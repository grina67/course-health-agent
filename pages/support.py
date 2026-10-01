# pages/support.py
#
# Сторінка «Центр підтримки» для Course Health Agent.
# Сюди агент веде користувача, якщо питання виходить за межі аналізу.
#
# УВАГА: усі контакти вигадані (демо для навчального проєкту).
# Номери 000-00-0x не існують, а домен example.com зарезервований для прикладів.

import streamlit as st

from agent import SUPPORTED_QUESTIONS

st.set_page_config(page_title="Підтримка · Course Health Agent", page_icon="💬", layout="wide")

SUPPORT = {
    "phone_main": "+380 (44) 000-00-01",
    "phone_mobile": "+380 (67) 000-00-02",
    "email": "help@example.com",
    "site": "https://support.example.com",
    "hours": "Пн–Пт: 9:00–19:00, Сб: 10:00–15:00 (за Києвом)",
}

st.page_link("app.py", label="Назад до дашборду", icon="⬅️")

st.title("💬 Центр підтримки")
st.markdown(
    "Привіт! 👋 Ми поруч, якщо агент не зміг відповісти на ваше питання або щось "
    "працює не так, як очікувалось. Оберіть зручний спосіб зв'язку — **відповімо якнайшвидше**."
)

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("### 📞 Подзвонити")
    st.markdown(f"**{SUPPORT['phone_main']}** — гаряча лінія")
    st.markdown(f"**{SUPPORT['phone_mobile']}** — мобільний")
    st.caption("Найшвидший спосіб, якщо питання термінове.")
with c2, st.container(border=True):
    st.markdown("### ✉️ Написати")
    st.markdown(f"[{SUPPORT['email']}](mailto:{SUPPORT['email']})")
    st.caption("Відповідаємо протягом одного робочого дня.")
with c3, st.container(border=True):
    st.markdown("### 🌐 Сайт підтримки")
    st.link_button("Відкрити сайт", SUPPORT["site"], width="stretch")
    st.caption("База знань, інструкції та онлайн-чат.")

st.info(f"🕘 **Графік роботи:** {SUPPORT['hours']}")

st.subheader("Щоб ми допомогли швидше")
st.markdown(
    "- **Опишіть, що саме хотіли дізнатися**: можна просто скопіювати своє питання агенту.\n"
    "- **Вкажіть активні фільтри** (domain, level), якщо вони були вибрані.\n"
    "- **Додайте скриншот**, якщо бачите помилку на сторінці.\n"
    "- Не надсилайте паролі, ключі доступу та персональні дані студентів."
)

with st.expander("❓ На що агент уже вміє відповідати сам"):
    st.markdown("\n".join(f"- {q}" for q in SUPPORTED_QUESTIONS))

st.caption("ℹ️ Контакти на цій сторінці демонстраційні (навчальний проєкт).")
