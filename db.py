# db.py
#
# Безпечне підключення до PostgreSQL та виконання дозволених запитів.
#
# - Рядок підключення береться лише зі змінної середовища DATABASE_URL
#   (файл .env або оточення); у коді його немає.
# - Кожен запит виконується в транзакції READ ONLY з обмеженням часу виконання.
# - Виконуються тільки запити з queries.ALLOWED_QUERIES за їхнім ключем.

import os
import re

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from queries import ALLOWED_QUERIES

load_dotenv()

STATEMENT_TIMEOUT_MS = 30_000

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|MERGE|CALL|DO)\b",
    re.IGNORECASE,
)

_engine: Engine | None = None


class DatabaseError(Exception):
    """Помилка підключення або запиту з повідомленням для користувача."""


def _short_reason(exc: Exception) -> str:
    """Перший рядок причини помилки без рядка підключення."""
    reason = str(getattr(exc, "orig", None) or exc).strip().splitlines()
    first = reason[0] if reason else exc.__class__.__name__
    # Прибираємо можливий URL з паролем, якщо драйвер його повторив.
    return re.sub(r"\w+(\+\w+)?://\S+", "<DATABASE_URL>", first)


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseError(
            "Не задано DATABASE_URL. Скопіюйте .env.example у .env "
            "і вкажіть рядок підключення до PostgreSQL."
        )
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Явно драйвер psycopg (v3) з requirements.txt: SQLAlchemy < 2.1 для
    # "postgresql://" за замовчуванням шукав би psycopg2.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]

    try:
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            # Без startup-параметра "options": пулери з'єднань (PgBouncer,
            # Supabase/Neon pooler) його не підтримують. Read-only і таймаут
            # вмикаються всередині кожної транзакції в run_query().
            connect_args={"connect_timeout": 10},
        )
    except (ArgumentError, ValueError) as exc:
        raise DatabaseError(
            "Некоректний формат DATABASE_URL. Очікується "
            "postgresql://<user>:<password>@<host>:<port>/<database>"
        ) from exc
    except ModuleNotFoundError as exc:
        raise DatabaseError(
            "Не встановлено драйвер PostgreSQL. Виконайте: pip install -r requirements.txt"
        ) from exc
    return _engine


def _check_read_only(sql: str) -> None:
    stripped = sql.lstrip().upper()
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        raise DatabaseError("Дозволені лише SELECT-запити.")
    if _FORBIDDEN_SQL.search(sql):
        raise DatabaseError("Запит містить заборонену операцію і не буде виконаний.")


def run_query(name: str) -> pd.DataFrame:
    """Виконує дозволений запит за ключем із queries.ALLOWED_QUERIES."""
    if name not in ALLOWED_QUERIES:
        raise DatabaseError(f"Запит '{name}' не входить до списку дозволених.")
    sql = ALLOWED_QUERIES[name]
    _check_read_only(sql)

    engine = get_engine()
    try:
        with engine.connect() as conn, conn.begin():
            # Перші команди транзакції: лише читання + обмеження часу.
            # SET LOCAL діє до кінця транзакції, тож сумісний з пулерами.
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
            return pd.read_sql_query(text(sql), conn)
    except SQLAlchemyError as exc:
        raise DatabaseError(
            "Не вдалося отримати дані з PostgreSQL: "
            f"{_short_reason(exc)}. Перевірте DATABASE_URL, доступність сервера "
            "та права користувача на читання таблиць enrollments і dim_course."
        ) from exc


def load_course_metrics() -> pd.DataFrame:
    """Метрики рівня курсу з числовими типами, готовими для pandas/Plotly."""
    df = run_query("course_metrics")
    numeric_cols = [
        "enrollment_count", "certified_count", "completion_rate", "avg_progress",
        "valid_progress_count", "invalid_progress_count",
        "p75_enrollments", "overall_completion_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_at_risk"] = df["is_at_risk"].astype(bool)
    for col in ["course", "partner", "domain", "level"]:
        df[col] = df[col].fillna("Unknown")
    return df
