# agent.py
#
# Аналітичний агент Course Health Agent.
#
# Як працює:
#   1. Питання користувача зіставляється з одним із підтримуваних сценаріїв
#      (детерміновано, за ключовими словами — без участі моделі).
#   2. Для сценарію з уже завантажених метрик рівня курсу (db.load_course_metrics)
#      будується невелика агрегована таблиця — у межах активних фільтрів
#      domain/level з app.py (без фільтрів — по всіх курсах). Пороги at-risk
#      при цьому лишаються глобальними.
#   3. Gemini отримує лише цю таблицю та загальні показники і формулює
#      відповідь українською. Модель не має інструментів, не бачить SQL
#      і не може нічого виконати в базі.
#   4. Якщо питання не підтримується, Gemini не викликається —
#      агент пояснює обмеження.

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Python за замовчуванням не бачить сертифікатів зі сховища Windows. Якщо
# антивірус (напр., Avast Web Shield) чи корпоративний проксі перехоплює HTTPS,
# запити до Gemini падають з CERTIFICATE_VERIFY_FAILED. truststore змушує Python
# довіряти системному сховищу; перевірка SSL при цьому лишається увімкненою.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

# Ланцюжок моделей (модель, thinking_level). Агенту треба лише стисло переказати
# готову таблицю, тож першими йдуть швидкі Flash-Lite (~1–2 с); повна Flash
# (~13–20 с, часто 503 під навантаженням) — останній резерв.
# Наступна модель пробується при 404 / 429 / 5xx / таймауті / порожній відповіді.
GEMINI_MODEL_CHAIN = (
    ("gemini-flash-lite-latest", None),
    ("gemini-3.5-flash-lite", None),
    ("gemini-flash-latest", "low"),
)
GEMINI_REQUEST_TIMEOUT_S = 20   # на один запит
GEMINI_TOTAL_BUDGET_S = 30      # на всі спроби разом
GEMINI_MIN_TIMEOUT_S = 10       # мінімальний дедлайн, який приймає Gemini API
GEMINI_HEDGE_AFTER_S = 3        # без відповіді за стільки секунд — дубль на наступну модель
GEMINI_MAX_OUTPUT_TOKENS = 700  # відповіді ~150–250 токенів; ліміт — запобіжник
GEMINI_CACHE_TTL_S = 600        # як кеш даних у app.py
GEMINI_CACHE_SIZE = 128
TOP_N = 10
MIN_ENROLLMENTS_IN_DOMAIN = 10

SUPPORTED_QUESTIONS = [
    "Які 10 найпопулярніших курсів мають completion rate нижче середнього?",
    "Порівняй completion rate курсів Beginner та Intermediate.",
    "Які курси домену Data Science мають найнижчий completion rate?",
    "Які популярні курси потребують уваги?",
    "У якого партнера найбільше at-risk курсів?",
]

# Колонки, які дозволено передавати в Gemini: лише агрегати рівня курсу/групи.
ALLOWED_PAYLOAD_COLUMNS = {
    "course", "partner", "domain", "level",
    "enrollment_count", "completion_rate", "avg_progress", "invalid_progress_count",
    "courses", "at_risk_courses", "at_risk_enrollments",
}

_PERSONAL_DATA_KEYWORDS = (
    "email", "e-mail", "пошт", "ім'я", "імена", "імен ", "прізвищ", "телефон",
    "user_id", "student_id", "персональн", "особист",
)

_LEVELS = ("Beginner", "Intermediate", "Advanced")

SYSTEM_INSTRUCTION = """Ти — аналітик освітньої платформи в застосунку Course Health Agent.
Правила:
- Відповідай українською мовою, стисло: 3–6 речень і, за потреби, короткий маркований список.
- Використовуй ЛИШЕ надані агреговані дані. Не вигадуй курсів, чисел чи причин.
- Відсотки подавай з одним знаком після коми; completion rate порівнюй із загальним.
- Не пиши SQL і не пропонуй змінювати базу даних.
- Якщо даних недостатньо для висновку, прямо скажи про це.
- Назви курсів, партнерів, доменів і рівнів залишай мовою оригіналу.
- Дані стосуються лише вибірки з поля selection. Сам опис вибірки вже показано
  над відповіддю — не повторюй його, але не узагальнюй висновки на всі курси.
- Поріг популярності (p75_enrollments) і загальний completion rate
  (overall_completion_rate) розраховані по всьому набору курсів."""

GLOSSARY = {
    "enrollment_count": "кількість реєстрацій на курс",
    "completion_rate": "AVG(is_certified) * 100, %",
    "avg_progress": "середній progress_pct лише для значень 0–100, %",
    "invalid_progress_count": "кількість значень progress_pct поза 0–100",
    "at_risk": "enrollment_count >= 75-го перцентиля і completion_rate < загального зваженого",
}


class GeminiError(Exception):
    """Помилка звернення до Gemini.

    str(exc) — зрозуміле повідомлення для користувача, без технічних деталей.
    kind    — тип: quota | auth | config | network | unavailable (для заголовка в UI).
    details — технічні деталі (моделі, HTTP-коди) лише для серверного логу;
              ключ і текст відповіді API сюди не потрапляють.
    """

    def __init__(self, message: str, kind: str = "unavailable", details: str = ""):
        super().__init__(message)
        self.kind = kind
        self.details = details


@dataclass
class AgentAnswer:
    text: str
    intent: str
    table: pd.DataFrame | None = None
    error: str | None = None
    facts: dict = field(default_factory=dict)
    scope: str = "усі курси"
    model: str | None = None
    suggest_support: bool = False  # показати в UI посилання на центр підтримки
    cached: bool = False           # відповідь взято з кешу (без запиту до Gemini)
    error_kind: str | None = None  # тип помилки Gemini (див. GeminiError.kind)


# ---------------------------------------------------------------------------
# Розпізнавання сценарію
# ---------------------------------------------------------------------------

def _match_domain(question: str, domains: list[str]) -> str | None:
    q = question.lower()
    found = [
        d for d in domains
        if d and d != "Unknown"
        and re.search(rf"(?<!\w){re.escape(d.lower())}(?!\w)", q)
    ]
    return max(found, key=len) if found else None


def detect_intent(question: str, df: pd.DataFrame) -> tuple[str, dict]:
    q = question.lower()

    if any(k in q for k in _PERSONAL_DATA_KEYWORDS):
        return "personal_data", {}

    if "партнер" in q or "partner" in q:
        return "partner_at_risk", {}

    levels = [lvl for lvl in _LEVELS if lvl.lower() in q]
    if levels and ("порівн" in q or "compare" in q or len(levels) >= 2):
        if len(levels) < 2:
            levels = ["Beginner", "Intermediate"]
        return "level_compare", {"levels": levels}

    domain = _match_domain(question, sorted(df["domain"].unique()))
    if domain:
        return "domain_lowest", {"domain": domain}
    if "домен" in q or "domain" in q:
        return "unknown_domain", {}

    if ("найпопулярн" in q or "топ" in q or "top" in q) and ("нижче" in q or "середн" in q):
        return "top_popular_below_avg", {}

    if any(k in q for k in ("уваг", "at-risk", "at risk", "ризик", "популярн")):
        return "at_risk", {}

    return "unsupported", {}


# ---------------------------------------------------------------------------
# Агреговані дані для кожного сценарію
# ---------------------------------------------------------------------------

FILTER_COLUMNS = ("domain", "level")


def apply_filters(df: pd.DataFrame, filters: dict[str, list[str]] | None) -> pd.DataFrame:
    """Відбирає курси за активними фільтрами; порожній список = без фільтра.

    Колонки is_at_risk, p75_enrollments і overall_completion_rate не
    перераховуються — визначення at-risk лишається глобальним.
    """
    for col in FILTER_COLUMNS:
        values = (filters or {}).get(col)
        if values:
            df = df[df[col].isin(values)]
    return df


def describe_scope(filters: dict[str, list[str]] | None) -> str:
    parts = [f"{col} = {', '.join(values)}"
             for col in FILTER_COLUMNS
             if (values := (filters or {}).get(col))]
    return "відфільтровані курси (" + "; ".join(parts) + ")" if parts else "усі курси"


def _global_facts(df: pd.DataFrame) -> dict:
    """Показники всього набору та глобальні пороги at-risk."""
    return {
        "all_courses": int(len(df)),
        "all_enrollments": int(df["enrollment_count"].sum()),
        "overall_completion_rate": round(float(df["overall_completion_rate"].iloc[0]), 2),
        "p75_enrollments": round(float(df["p75_enrollments"].iloc[0]), 1),
        "at_risk_courses_all": int(df["is_at_risk"].sum()),
    }


def _selection_facts(sub: pd.DataFrame, scope: str) -> dict:
    """Показники поточної вибірки (може збігатися з усім набором)."""
    enrollments = int(sub["enrollment_count"].sum())
    return {
        "selection": scope,
        "selection_courses": int(len(sub)),
        "selection_enrollments": enrollments,
        "selection_completion_rate": (
            round(float(sub["certified_count"].sum() * 100 / enrollments), 2)
            if enrollments else None),
        "selection_at_risk_courses": int(sub["is_at_risk"].sum()),
    }


def _group_metrics(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Зважені метрики групи курсів (рівень, партнер тощо)."""
    tmp = df.assign(progress_sum=df["avg_progress"].fillna(0) * df["valid_progress_count"])
    g = tmp.groupby(by).agg(
        courses=("course_id", "count"),
        enrollment_count=("enrollment_count", "sum"),
        certified=("certified_count", "sum"),
        progress_sum=("progress_sum", "sum"),
        valid=("valid_progress_count", "sum"),
        invalid_progress_count=("invalid_progress_count", "sum"),
        at_risk_courses=("is_at_risk", "sum"),
    )
    g["completion_rate"] = g["certified"] * 100 / g["enrollment_count"]
    g["avg_progress"] = g["progress_sum"] / g["valid"].where(g["valid"] > 0)
    return g.drop(columns=["certified", "progress_sum", "valid"]).reset_index()


COURSE_COLUMNS = ["course", "partner", "domain", "level",
                  "enrollment_count", "completion_rate", "avg_progress", "invalid_progress_count"]


def build_context(intent: str, params: dict, df: pd.DataFrame) -> tuple[str, pd.DataFrame, dict]:
    """Повертає (опис вибірки, агрегована таблиця, додаткові факти)."""
    overall = float(df["overall_completion_rate"].iloc[0])

    if intent == "top_popular_below_avg":
        table = (df[df["completion_rate"] < overall]
                 .nlargest(TOP_N, "enrollment_count")[COURSE_COLUMNS])
        note = (f"Топ-{TOP_N} курсів за кількістю реєстрацій серед курсів, "
                "у яких completion_rate нижчий за загальний зважений.")
        return note, table, {}

    if intent == "level_compare":
        levels = params["levels"]
        table = _group_metrics(df[df["level"].isin(levels)], "level")
        table = table[["level", "courses", "enrollment_count", "completion_rate",
                       "avg_progress", "invalid_progress_count", "at_risk_courses"]]
        note = ("Порівняння рівнів: completion_rate і avg_progress зважені "
                "за кількістю реєстрацій усіх курсів рівня.")
        return note, table, {"levels": levels}

    if intent == "domain_lowest":
        domain = params["domain"]
        sub = df[df["domain"] == domain]
        if sub.empty:  # домен є в базі, але не входить у поточну вибірку
            return f"Домен {domain} відсутній у поточній вибірці.", sub[COURSE_COLUMNS], {}
        eligible = sub[sub["enrollment_count"] >= MIN_ENROLLMENTS_IN_DOMAIN]
        threshold = MIN_ENROLLMENTS_IN_DOMAIN
        if len(eligible) < 3:
            eligible, threshold = sub, 1
        table = (eligible.sort_values(["completion_rate", "enrollment_count"],
                                      ascending=[True, False])
                 .head(TOP_N)[COURSE_COLUMNS])
        domain_rate = sub["certified_count"].sum() * 100 / sub["enrollment_count"].sum()
        note = (f"Курси домену {domain} з найнижчим completion_rate "
                f"(враховано курси з ≥ {threshold} реєстраціями).")
        return note, table, {
            "domain": domain,
            "domain_courses": int(len(sub)),
            "domain_enrollments": int(sub["enrollment_count"].sum()),
            "domain_completion_rate": round(float(domain_rate), 2),
            "min_enrollments_used": threshold,
        }

    if intent == "at_risk":
        table = df[df["is_at_risk"]].nlargest(TOP_N, "enrollment_count")[COURSE_COLUMNS]
        note = (f"Топ-{TOP_N} at-risk курсів за кількістю реєстрацій "
                "(популярні курси з completion_rate нижче загального).")
        return note, table, {}

    if intent == "partner_at_risk":
        at_risk = df[df["is_at_risk"]]
        grouped = _group_metrics(at_risk, "partner").rename(
            columns={"enrollment_count": "at_risk_enrollments"})
        table = (grouped.sort_values(["at_risk_courses", "at_risk_enrollments"],
                                     ascending=False)
                 .head(TOP_N)[["partner", "at_risk_courses", "at_risk_enrollments",
                               "completion_rate", "avg_progress"]])
        note = ("Партнери з найбільшою кількістю at-risk курсів; completion_rate "
                "і avg_progress — зважені по їхніх at-risk курсах.")
        return note, table, {}

    raise ValueError(f"Невідомий сценарій: {intent}")


def _payload(table: pd.DataFrame) -> list[dict]:
    extra = set(table.columns) - ALLOWED_PAYLOAD_COLUMNS
    if extra:
        # Захист від випадкової передачі непередбачених колонок у модель.
        raise ValueError(f"Колонки не дозволені для передачі в Gemini: {sorted(extra)}")
    return json.loads(table.round(2).to_json(orient="records", force_ascii=False))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class _ModelFailed(Exception):
    """Збій однієї моделі — можна пробувати наступну."""


_client = None
_client_key = None
_client_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")
_cache: OrderedDict[str, tuple[float, str, str]] = OrderedDict()
_cache_lock = threading.Lock()


def _get_client(api_key: str):
    """Один клієнт Gemini на процес: з'єднання (TLS) перевикористовується між питаннями."""
    global _client, _client_key
    from google import genai
    from google.genai import types

    with _client_lock:
        if _client is None or _client_key != api_key:
            # Одна спроба на запит: власні повтори SDK (до 5 спроб з паузами до 60 с)
            # і відсутність таймауту раніше давали «зависання» на хвилини.
            _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))
            _client_key = api_key
        return _client


def _call_model(client, model: str, thinking_level: str | None,
                prompt: str, timeout_s: float) -> str:
    """Один запит до однієї моделі. GeminiError — фатально, _ModelFailed — пробуємо іншу."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        # У «думаючих» моделей токени міркувань входять у ліміт — їм ліміт не ставимо.
        max_output_tokens=None if thinking_level else GEMINI_MAX_OUTPUT_TOKENS,
        # Інструментів у моделі немає; AFC вимкнено явно.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=(types.ThinkingConfig(thinking_level=thinking_level)
                         if thinking_level else None),
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000),
                                       retry_options=types.HttpRetryOptions(attempts=1)),
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except errors.APIError as exc:
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or "")
        if code in (401, 403) or (code == 400 and "api key" in message.lower()):
            raise GeminiError(
                "Ключ Gemini недійсний або не має доступу до Gemini API. "
                "Перевірте значення GEMINI_API_KEY у файлі .env.",
                kind="auth", details=f"{model}: HTTP {code}",
            ) from exc
        # Лише код і коротка причина — текст відповіді API не зберігаємо й не показуємо.
        reason = {404: "модель недоступна", 429: "ліміт запитів/квота",
                  400: "некоректні параметри для моделі"}.get(
            code, "перевантажена або недоступна" if (code or 0) >= 500 else "інша помилка API")
        raise _ModelFailed(f"{model}: {code} {reason}") from exc
    except Exception as exc:  # таймаут, мережа, SSL
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise GeminiError(
                "Не вдалося встановити захищене з'єднання з Gemini. Ймовірно, антивірус "
                "або проксі перехоплює HTTPS — переконайтеся, що встановлено залежності "
                "з requirements.txt (пакет truststore).",
                kind="network", details="SSL CERTIFICATE_VERIFY_FAILED",
            ) from exc
        if "timeout" in type(exc).__name__.lower():
            raise _ModelFailed(f"{model}: таймаут ({timeout_s:.0f} с)") from exc
        raise _ModelFailed(f"{model}: {type(exc).__name__}") from exc

    text = (response.text or "").strip()
    if not text:
        raise _ModelFailed(f"{model}: порожня відповідь")
    return text


def _cache_get(key: str) -> tuple[str, str] | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < GEMINI_CACHE_TTL_S:
            _cache.move_to_end(key)
            return hit[1], hit[2]
        _cache.pop(key, None)
        return None


def _cache_put(key: str, text: str, model: str) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), text, model)
        _cache.move_to_end(key)
        while len(_cache) > GEMINI_CACHE_SIZE:
            _cache.popitem(last=False)


def ask_gemini(prompt: str) -> tuple[str, str, bool]:
    """Повертає (текст відповіді, модель, чи взято з кешу).

    Прискорення:
      * кеш: той самий промпт (питання + вибірка + дані) → миттєва відповідь;
      * «дублювання» запиту (hedging): якщо модель не відповіла за
        GEMINI_HEDGE_AFTER_S, паралельно запускається наступна, і береться перша
        успішна відповідь. Gemini інколи тримає запит у черзі 10–20 с, хоча
        зазвичай відповідає за 1–2 с, — дубль прибирає ці «хвости».
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiError(
            "Ключ Gemini не налаштовано: додайте GEMINI_API_KEY у файл .env "
            "(див. .env.example).",
            kind="config", details="GEMINI_API_KEY is empty",
        )
    try:
        client = _get_client(api_key)
    except ImportError as exc:
        raise GeminiError(
            "Не встановлено бібліотеку google-genai. Виконайте: "
            "pip install -r requirements.txt",
            kind="config", details="google-genai import failed",
        ) from exc

    chain = list(GEMINI_MODEL_CHAIN)
    override = os.getenv("GEMINI_MODEL", "").strip()
    if override:
        chain = [(override, None)] + [c for c in chain if c[0] != override]

    cache_key = hashlib.sha256(
        f"{SYSTEM_INSTRUCTION}\x00{chain}\x00{prompt}".encode("utf-8")).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return cached[0], cached[1], True

    started = time.monotonic()
    failures: list[str] = []
    pending = iter(chain)
    running: dict = {}

    def launch_next() -> bool:
        remaining = GEMINI_TOTAL_BUDGET_S - (time.monotonic() - started)
        # Сервер Gemini відхиляє дедлайн коротший за 10 с (помилка 400).
        if remaining < GEMINI_MIN_TIMEOUT_S:
            return False
        nxt = next(pending, None)
        if nxt is None:
            return False
        model, thinking_level = nxt
        timeout_s = min(GEMINI_REQUEST_TIMEOUT_S, remaining)
        running[_executor.submit(_call_model, client, model, thinking_level,
                                 prompt, timeout_s)] = model
        return True

    launch_next()
    while running:
        remaining = GEMINI_TOTAL_BUDGET_S - (time.monotonic() - started)
        if remaining <= 0:
            break
        done, _ = wait(running, timeout=min(GEMINI_HEDGE_AFTER_S, remaining),
                       return_when=FIRST_COMPLETED)
        if not done:
            launch_next()  # відповіді досі немає — дублюємо запит на наступну модель
            continue
        for fut in done:
            model = running.pop(fut)
            try:
                text = fut.result()
            except _ModelFailed as exc:
                failures.append(str(exc))
                continue
            # Решта запитів завершиться у фоні за власним таймаутом; їх ігноруємо.
            _cache_put(cache_key, text, model)
            return text, model, False
        if not running:
            launch_next()  # усі запущені впали — пробуємо наступну модель одразу

    if running:
        failures.extend(f"{m}: не встиг за {GEMINI_TOTAL_BUDGET_S} с" for m in running.values())
    details = "; ".join(failures) + f" ({time.monotonic() - started:.0f} s)"
    if any(" 429 " in f for f in failures):
        raise GeminiError(
            "Безкоштовний ліміт запитів Gemini тимчасово вичерпано. Зачекайте 1–2 хвилини "
            "й спробуйте знову; якщо не допоможе — денний ліміт оновиться пізніше. "
            "Дашборд працює як звичайно, а вже поставлені питання відповідають з кешу.",
            kind="quota", details=details,
        )
    raise GeminiError(
        "Gemini зараз перевантажений або недоступний, тому агент не зміг сформулювати "
        "відповідь. Спробуйте ще раз за хвилину — дашборд працює як звичайно.",
        kind="unavailable", details=details,
    )


def _build_prompt(question: str, note: str, facts: dict, rows: list[dict]) -> str:
    data = {
        "загальні_показники": facts,
        "визначення_метрик": GLOSSARY,
        "опис_вибірки": note,
        "дані": rows,
    }
    return (
        f"Питання користувача: {question}\n\n"
        "Агреговані дані рівня курсів (JSON):\n"
        f"{json.dumps(data, ensure_ascii=False, indent=1)}\n\n"
        "Дай відповідь на питання, спираючись лише на ці дані."
    )


# ---------------------------------------------------------------------------
# Публічна функція
# ---------------------------------------------------------------------------

def _out_of_scope_text() -> str:
    examples = "\n".join(f"    - {q}" for q in SUPPORTED_QUESTIONS)  # вкладений список
    return (
        "🤔 **На жаль, з цим питанням я допомогти не можу.** Я аналітичний агент "
        "і вмію відповідати лише на питання про популярність і завершення курсів: "
        "кількість реєстрацій, completion rate, середній прогрес і курси, які потребують уваги.\n\n"
        "**Що можна зробити:**\n"
        "1. Переформулювати питання або вибрати одне з тих, на які я точно відповім:\n"
        f"{examples}\n"
        "2. Звернутися до нашої **команди підтримки**: там охоче допоможуть з будь-яким "
        "іншим запитом. Контакти — за кнопкою «Зв'язатися з підтримкою» вище. ☝️"
    )


def _limitations_text() -> str:
    examples = "\n".join(f"- {q}" for q in SUPPORTED_QUESTIONS)
    return (
        "Агент аналізує лише популярність і завершення курсів на основі агрегованих "
        "метрик (реєстрації, completion rate, середній прогрес, at-risk курси). "
        "Він не виконує довільних SQL-запитів і не працює з даними окремих студентів.\n\n"
        f"Спробуйте одне з підтримуваних питань:\n{examples}"
    )


def answer_question(question: str, df: pd.DataFrame,
                    filters: dict[str, list[str]] | None = None) -> AgentAnswer:
    """Відповідає на питання в межах вибірки, заданої фільтрами.

    df — метрики ВСІХ курсів (глобальні пороги at-risk уже пораховані в SQL);
    filters — активні фільтри з app.py, напр. {"domain": [...], "level": [...]}.
    """
    question = (question or "").strip()
    if not question:
        return AgentAnswer("Введіть питання.", "empty")

    scope = describe_scope(filters)
    sub = apply_filters(df, filters)

    def with_scope(body: str) -> str:
        return f"**Вибірка:** {scope}.\n\n{body}"

    # Домен розпізнаємо по всьому набору, щоб відрізнити «домену немає в базі»
    # від «домен відфільтровано».
    intent, params = detect_intent(question, df)

    if intent == "personal_data":
        return AgentAnswer(
            "Агент не працює з персональними даними (імена, email, ідентифікатори "
            "студентів тощо) — доступні лише агреговані метрики рівня курсів.\n\n"
            + _limitations_text(),
            intent, scope=scope,
        )
    if intent == "unknown_domain":
        top_domains = ", ".join(
            df.groupby("domain")["enrollment_count"].sum().nlargest(8).index)
        return AgentAnswer(
            "Не вдалося знайти вказаний домен у даних. Назвіть домен так, як він "
            f"записаний у базі, наприклад: {top_domains}.",
            intent, scope=scope,
        )
    if intent == "unsupported":
        return AgentAnswer(_out_of_scope_text(), intent, scope=scope, suggest_support=True)

    facts = {**_global_facts(df), **_selection_facts(sub, scope)}
    if sub.empty:
        return AgentAnswer(with_scope("У поточній вибірці немає курсів. Змініть фільтри."),
                           intent, facts=facts, scope=scope)

    note, table, extra_facts = build_context(intent, params, sub)
    facts.update(extra_facts)

    if table.empty:
        return AgentAnswer(
            with_scope("У цій вибірці немає курсів, що відповідають умовам питання. "
                       "Спробуйте змінити або скинути фільтри."),
            intent, table, facts=facts, scope=scope)

    prompt = _build_prompt(question, note, facts, _payload(table))
    try:
        text, model, cached = ask_gemini(prompt)
        return AgentAnswer(with_scope(text), intent, table, facts=facts, scope=scope,
                           model=model, cached=cached)
    except GeminiError as exc:
        # Технічні деталі — лише в серверний лог; користувач бачить str(exc).
        logger.warning("Gemini недоступний [%s]: %s", exc.kind, exc.details)
        fallback = (f"{note}\n\nВідповідь моделі недоступна, тому нижче показано "
                    "дані, на яких вона ґрунтувалася б.")
        return AgentAnswer(with_scope(fallback), intent, table, error=str(exc),
                           error_kind=exc.kind, facts=facts, scope=scope)
