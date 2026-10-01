# app.py
#
# Streamlit-інтерфейс проєкту "Course Health Agent".
# Домашнє завдання: Database -> Agent -> Analysis -> Dashboard.
#
# Компонування (щоб основне вміщалося на одному екрані):
#   заголовок
#   [ KPI-плитки стовпчиком ] [ агент: приклади, поле, відповідь ] [ scatter-діаграма ]
#   Top-10 at-risk courses (нижче — максимум один скрол)
#
# Запуск: streamlit run app.py

import html
import math
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import SUPPORTED_QUESTIONS, answer_question, apply_filters, describe_scope
from db import DatabaseError, load_course_metrics

# initial_sidebar_state=230 — штатна ширина бічної панелі в пікселях.
# (CSS-перевизначення ширини з !important ламало кнопку «Clear all» у фільтрах.)
st.set_page_config(page_title="Course Health Agent", page_icon="📊", layout="wide",
                   initial_sidebar_state=230)

# Менші відступи сторінки, щоб усе основне вміщалося на екрані.
st.markdown(
    """<style>
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 2.2rem; padding-bottom: 1rem;
    }
    h2 { padding-top: 0 !important; padding-bottom: 0.1rem !important; }


    /* Вбудована кнопка «Clear all» у фільтрах: у вузькій бічній панелі клік по ній
       часто не спрацьовує (особливість компонента Streamlit) — ховаємо її, замість неї
       кнопка «↺ Скинути фільтри»; окремі значення знімаються хрестиком на тезі. */
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] button[aria-label="Clear all"] {
        display: none;
    }

    /* KPI — маленькі квадратні плитки в стовпчик. */
    .st-key-kpi_tiles { gap: 0.6rem; }
    /* Фіксована ширина стовпчика KPI — щоб на вузьких екранах числа не обрізалися. */
    [data-testid="stColumn"]:has(.st-key-kpi_tiles) {
        flex: 0 0 136px !important; width: 136px !important; min-width: 136px !important;
    }
    /* Ряд KPI | агент | діаграма не переноситься на вузьких екранах (≥ 1280 px). */
    @media (min-width: 1024px) {
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-kpi_tiles) {
            flex-wrap: nowrap !important;
        }
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-kpi_tiles)
            > [data-testid="stColumn"]:not(:has(.st-key-kpi_tiles)) {
            min-width: 0 !important;
        }
    }
    .st-key-kpi_tiles [data-testid="stMetric"] {
        aspect-ratio: 1 / 1;
        display: flex; flex-direction: column; justify-content: center;
        padding: 0.4rem 0.5rem; text-align: center;
    }
    .st-key-kpi_tiles [data-testid="stMetricLabel"] { justify-content: center; }
    .st-key-kpi_tiles [data-testid="stMetricLabel"] p { font-size: 0.78rem; white-space: nowrap; }
    .st-key-kpi_tiles [data-testid="stMetricValue"] { font-size: 1.35rem; justify-content: center; }
    .st-key-kpi_tiles [data-testid="stMetricDelta"] { justify-content: center; }
    .st-key-kpi_tiles [data-testid="stMetricDelta"] * {
        font-size: 0.72rem; white-space: nowrap; overflow: visible; text-overflow: clip;
    }

    /* Питання користувача — дрібно й приглушено. */
    .user-question { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.3rem; }

    /* Відповідь агента — виділена картка: фон, акцентна смуга, інший шрифт.
       Напівпрозорі кольори виглядають коректно і в світлій, і в темній темі. */
    .st-key-agent_answer {
        background: rgba(28, 131, 225, 0.08);
        border-left: 4px solid #1c83e1;
        border-radius: 0.5rem;
        padding: 0.7rem 0.9rem 0.4rem;
        gap: 0.25rem;
    }
    .st-key-agent_answer p, .st-key-agent_answer li {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 0.97rem;
        line-height: 1.55;
    }
    .st-key-agent_answer strong { color: #1c83e1; }
    </style>""",
    unsafe_allow_html=True,
)

AGENT_PANEL_HEIGHT = 440   # висота блоку відповіді (далі — прокрутка всередині)
CHART_HEIGHT = 560         # ≈ висота колонки агента, щоб ряд був рівним

# Заголовки попереджень для типів помилок Gemini (agent.GeminiError.kind).
GEMINI_ERROR_TITLES = {
    "quota": ("⏳", "Ліміт Gemini тимчасово вичерпано."),
    "auth": ("🔑", "Проблема з ключем Gemini."),
    "config": ("⚙️", "Агента не налаштовано."),
    "network": ("🌐", "Немає з'єднання з Gemini."),
    "unavailable": ("⚠️", "Агент тимчасово недоступний."),
}

# Короткі підписи кнопок-прикладів → повні підтримувані питання.
EXAMPLE_LABELS = [
    "🔟 Топ-10 < середнього",
    "⚖️ Beginner vs Intermediate",
    "🔬 Data Science",
    "⚠️ Потребують уваги",
    "🏛️ Партнери at-risk",
]


@st.cache_data(ttl=600, show_spinner="Завантаження метрик курсів з PostgreSQL…")
def get_data() -> pd.DataFrame:
    return load_course_metrics()


def weighted_completion_rate(df: pd.DataFrame) -> float:
    total = df["enrollment_count"].sum()
    return float(df["certified_count"].sum() * 100 / total) if total else 0.0


def reset_filters() -> None:
    st.session_state["f_domain"] = []
    st.session_state["f_level"] = []


def pick_example(question: str) -> None:
    # Клік по прикладу одразу ставить питання агенту (поле вводу лишається порожнім).
    st.session_state["pending_question"] = question


# --- Заголовок --------------------------------------------------------------

st.markdown("## 📊 Course Health Agent")
st.caption(
    "Популярність і завершення курсів: реєстрації, completion rate і **популярні курси "
    "з низьким завершенням (at-risk)**. Лише агрегати рівня курсів, без персональних даних."
)

# --- Дані ------------------------------------------------------------------

try:
    data = get_data()
except DatabaseError as exc:
    st.error(f"**Помилка підключення до бази даних.** {exc}")
    st.page_link("pages/support.py", label="Не вдається вирішити? Зверніться до підтримки",
                 icon="💬")
    st.stop()

if data.empty:
    st.warning("Запит до бази не повернув жодного курсу.")
    st.stop()

overall_rate = float(data["overall_completion_rate"].iloc[0])
p75 = float(data["p75_enrollments"].iloc[0])

# --- Бічна панель: фільтри + підтримка ---------------------------------------

with st.sidebar:
    st.header("Фільтри")
    domains = st.multiselect("Domain", sorted(data["domain"].unique()),
                             placeholder="Усі домени", key="f_domain")
    levels = st.multiselect("Level", sorted(data["level"].unique()),
                            placeholder="Усі рівні", key="f_level")
    # Власна кнопка скидання: вбудований «Clear all» у вузькій панелі
    # спрацьовує ненадійно (див. CSS вище); окремі значення знімаються хрестиком.
    st.button("↺ Скинути фільтри", on_click=reset_filters, width="stretch",
              disabled=not (domains or levels))
    st.divider()
    st.page_link("pages/support.py", label="Підтримка", icon="💬")
    st.divider()
    st.caption(
        f"Пороги at-risk розраховані по всіх курсах: enrollment_count ≥ {p75:.1f} "
        f"(75-й перцентиль) і completion_rate < {overall_rate:.2f}% "
        "(загальний зважений). Фільтри звужують вибірку для дашборду й агента, "
        "але не змінюють ці пороги."
    )

active_filters = {"domain": domains, "level": levels}
filtered = apply_filters(data, active_filters)

if filtered.empty:
    st.info("Немає курсів для вибраних фільтрів.")
    st.stop()

# --- KPI (стовпчик) | агент | діаграма ----------------------------------------

col_kpi, col_agent, col_chart = st.columns([1.15, 5.6, 5.6], gap="medium")

with col_kpi:
    # Компактні квадратні плитки, складені в стовпчик (стилі .st-key-kpi_tiles угорі).
    is_filtered = bool(domains or levels)
    enr_sel = int(filtered["enrollment_count"].sum())
    enr_all = int(data["enrollment_count"].sum())
    cert_sel = int(filtered["certified_count"].sum())
    filtered_rate = weighted_completion_rate(filtered)
    popular_sel = int((filtered["enrollment_count"] >= p75).sum())
    at_risk_sel = int(filtered["is_at_risk"].sum())

    def fmt(n: float) -> str:
        return f"{n:,.0f}".replace(",", " ")

    # Підказки (значок «?») — простою мовою, з числами поточної вибірки.
    help_enr = (
        "**Скільки разів записувалися на курси** у поточній вибірці.\n\n"
        "Одна людина може бути записана на кілька курсів, тож це **не кількість студентів**.\n\n"
        + (f"Це {enr_sel / enr_all:.1%} від усіх {fmt(enr_all)} реєстрацій."
           if is_filtered else f"Усього {fmt(len(data))} курсів, на які є хоча б один запис.")
    )
    one_in = f"≈ 1 з {round(100 / filtered_rate)}" if filtered_rate else "жоден із"
    help_compl = (
        "**Яка частка реєстрацій закінчилася сертифікатом.**\n\n"
        f"{fmt(cert_sel)} сертифікатів ÷ {fmt(enr_sel)} реєстрацій = **{filtered_rate:.2f}%**, "
        f"тобто {one_in} записів доходить до кінця.\n\n"
        f"Середній показник по всіх курсах — **{overall_rate:.2f}%** "
        "(червона пунктирна лінія на діаграмі)."
        + ("\n\nСтрілка під числом — на скільки процентних пунктів (пп) вибірка краща ↑ "
           "або гірша ↓ за цей середній показник." if is_filtered else "")
    )
    help_risk = (
        "**Популярні курси, які погано завершують** — кандидати на покращення.\n\n"
        f"Курс потрапляє сюди, якщо:\n"
        f"- має **≥ {p75:.0f} реєстрацій** (25% найпопулярніших курсів), **і**\n"
        f"- його completion **нижчий за {overall_rate:.2f}%** (середній по всіх курсах).\n\n"
        f"У вибірці: **{at_risk_sel} з {popular_sel}** популярних курсів"
        + (f" ({at_risk_sel / popular_sel:.0%})" if popular_sel else "")
        + ".\n\nНайбільші з них — у таблиці **Top-10 at-risk courses** нижче."
    )

    with st.container(key="kpi_tiles"):
        st.metric("Реєстрації", fmt(enr_sel), help=help_enr, border=True)
        st.metric(
            "Completion",
            f"{filtered_rate:.2f}%",
            delta=f"{filtered_rate - overall_rate:+.2f} пп" if is_filtered else None,
            help=help_compl,
            border=True,
        )
        st.metric("At-risk", at_risk_sel, help=help_risk, border=True)

with col_agent:
    st.markdown("#### 🤖 Запитайте агента")
    st.caption(f"Вибірка: **{describe_scope(active_filters)}**. Оберіть приклад або напишіть своє:")

    # Кнопки-приклади в рядок із переносом; повне питання — у підказці при наведенні.
    with st.container(horizontal=True, wrap=True, gap="small"):
        for i, (label, full) in enumerate(zip(EXAMPLE_LABELS, SUPPORTED_QUESTIONS)):
            st.button(label, help=full, key=f"example_{i}", width="content",
                      on_click=pick_example, args=(full,))

    # Форма: і Enter у полі, і кнопка надсилають питання.
    # clear_on_submit — поле очищається одразу після надсилання.
    with st.form("ask_agent", border=False, clear_on_submit=True):
        f1, f2 = st.columns([5, 2], vertical_alignment="bottom")
        typed = f1.text_input("Ваше питання", key="question", label_visibility="collapsed",
                              placeholder="Ваше питання, напр.: Які курси потребують уваги?")
        submitted = f2.form_submit_button("Запитати", type="primary", width="stretch")

    answer_box = st.container(height=AGENT_PANEL_HEIGHT, border=True)

    question = st.session_state.pop("pending_question", None) or (typed if submitted else None)
    if submitted and not (question or "").strip():
        answer_box.warning("Введіть питання або виберіть приклад вище.")
    elif question:
        started = time.monotonic()
        with answer_box, st.spinner("Агент аналізує дані… (зазвичай кілька секунд)"):
            result = answer_question(question, data, active_filters)
        # Зберігаємо, щоб відповідь не зникала після зміни фільтрів чи інших дій.
        st.session_state["last_answer"] = (question, result, time.monotonic() - started)

    with answer_box:
        if "last_answer" in st.session_state:
            asked, result, elapsed = st.session_state["last_answer"]
            st.markdown(f'<div class="user-question">🙋 <b>Ваше питання:</b> {html.escape(asked)}</div>',
                        unsafe_allow_html=True)
            if result.error:
                icon, title = GEMINI_ERROR_TITLES.get(
                    result.error_kind, GEMINI_ERROR_TITLES["unavailable"])
                st.warning(f"**{title}** {result.error}", icon=icon)
            if result.suggest_support:
                st.page_link("pages/support.py", label="Зв'язатися з підтримкою", icon="💬")
            # Відповідь агента — окрема «картка» (стилі .st-key-agent_answer угорі файлу).
            with st.container(key="agent_answer"):
                st.markdown("**🤖 Відповідь агента**")
                st.markdown(result.text)
            if result.model:
                source = "з кешу" if result.cached else f"{elapsed:.1f} с"
                st.caption(f"Модель: {result.model} · {source}")
            if result.table is not None and not result.table.empty:
                with st.expander("Дані, передані агенту", expanded=result.error is not None):
                    st.dataframe(result.table.round(2), hide_index=True, width="stretch")
        else:
            st.caption("💡 Тут з'явиться відповідь агента. Клікніть на будь-який приклад "
                       "вище — відповідь займе кілька секунд.")

with col_chart:
    st.markdown("#### Популярність vs завершення")
    plot_df = filtered.assign(
        status=filtered["is_at_risk"].map({True: "At-risk", False: "Інші"}))
    fig = px.scatter(
        plot_df,
        x="enrollment_count",
        y="completion_rate",
        color="avg_progress",
        symbol="status",
        symbol_map={"At-risk": "diamond", "Інші": "circle"},
        log_x=True,
        color_continuous_scale="Viridis",
        hover_name="course",
        hover_data={
            "partner": True, "domain": True, "level": True,
            "enrollment_count": True,
            "completion_rate": ":.1f", "avg_progress": ":.1f",
            "status": False,
        },
        labels={
            "enrollment_count": "Реєстрацій (log)",
            "completion_rate": "Completion rate, %",
            "avg_progress": "Сер. прогрес, %",
        },
        height=CHART_HEIGHT,
    )
    fig.add_hline(
        y=overall_rate, line_dash="dash", line_color="red",
        annotation_text=f"Загальний completion rate {overall_rate:.2f}%",
        annotation_position="top left",
    )
    # Лінія P75 без вбудованого підпису: на лог-осі анотації задаються в log10,
    # і add_vline(annotation_text=...) розтягував вісь X до 10^30.
    fig.add_vline(x=p75, line_dash="dot", line_color="gray")
    fig.add_annotation(x=math.log10(p75), y=1, yref="paper", text=f"P75 = {p75:.0f}",
                       showarrow=False, xanchor="left", yanchor="top",
                       font={"color": "gray", "size": 11})
    fig.update_traces(marker={"opacity": 0.75})
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom", "title": None},
        coloraxis_colorbar={"title": "Прогрес, %", "thickness": 12},
    )
    st.plotly_chart(fig, width="stretch")

invalid_total = int(filtered["invalid_progress_count"].sum())
if invalid_total:
    st.caption(
        f"⚠️ {invalid_total} значень progress_pct поза межами 0–100 не враховано в avg_progress "
        "(дані в базі не змінювались)."
    )

# --- Top-10 at-risk --------------------------------------------------------

st.markdown("#### Top-10 at-risk courses")
top_risk = (filtered[filtered["is_at_risk"]]
            .nlargest(10, "enrollment_count")
            [["course", "partner", "domain", "level",
              "enrollment_count", "completion_rate", "avg_progress"]])
if top_risk.empty:
    st.info("Серед вибраних курсів немає at-risk.")
else:
    st.dataframe(
        top_risk,
        hide_index=True,
        width="stretch",
        column_config={
            "completion_rate": st.column_config.NumberColumn("completion_rate, %", format="%.1f"),
            "avg_progress": st.column_config.NumberColumn("avg_progress, %", format="%.1f"),
        },
    )
