# queries.py
#
# Єдине місце, де зберігаються SQL-запити проєкту.
# Тут дозволені лише заздалегідь визначені SELECT-запити (з CTE).
# Модель Gemini SQL не створює і не виконує: db.py запускає запит тільки
# за ключем із ALLOWED_QUERIES.
#
# Використовуються лише таблиці enrollments і dim_course.
# Запити повертають агрегати рівня курсу — без user_id та персональних даних.

# Метрики кожного курсу + пороги at-risk.
#   enrollment_count       — COUNT(*) реєстрацій курсу
#   certified_count        — SUM(is_certified), потрібен для зваженого rate
#   completion_rate        — AVG(is_certified) * 100
#   avg_progress           — AVG(progress_pct) лише для значень у межах 0–100
#   valid_progress_count   — кількість значень progress_pct у межах 0–100
#   invalid_progress_count — кількість значень progress_pct поза межами 0–100
#   p75_enrollments        — 75-й перцентиль enrollment_count серед усіх курсів
#   overall_completion_rate— загальний зважений rate: SUM(certified) / SUM(enrollments) * 100
#   is_at_risk             — enrollment_count >= p75 і completion_rate < overall
COURSE_METRICS_SQL = """
WITH course_stats AS (
    SELECT
        d.course_id,
        d.course,
        d.partner,
        d.domain,
        d.level,
        COUNT(*)                                            AS enrollment_count,
        SUM(e.is_certified)                                 AS certified_count,
        AVG(e.is_certified) * 100                           AS completion_rate,
        AVG(e.progress_pct)
            FILTER (WHERE e.progress_pct BETWEEN 0 AND 100) AS avg_progress,
        COUNT(*)
            FILTER (WHERE e.progress_pct BETWEEN 0 AND 100) AS valid_progress_count,
        COUNT(*)
            FILTER (WHERE e.progress_pct < 0
                       OR e.progress_pct > 100)             AS invalid_progress_count
    FROM enrollments AS e
    JOIN dim_course AS d ON d.course_id = e.course_id
    GROUP BY d.course_id, d.course, d.partner, d.domain, d.level
),
thresholds AS (
    SELECT
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY enrollment_count) AS p75_enrollments,
        SUM(certified_count) * 100.0 / SUM(enrollment_count)          AS overall_completion_rate
    FROM course_stats
)
SELECT
    cs.course_id,
    cs.course,
    cs.partner,
    cs.domain,
    cs.level,
    cs.enrollment_count,
    cs.certified_count,
    cs.completion_rate,
    cs.avg_progress,
    cs.valid_progress_count,
    cs.invalid_progress_count,
    t.p75_enrollments,
    t.overall_completion_rate,
    (cs.enrollment_count >= t.p75_enrollments
     AND cs.completion_rate < t.overall_completion_rate) AS is_at_risk
FROM course_stats AS cs
CROSS JOIN thresholds AS t
ORDER BY cs.enrollment_count DESC
"""

# Реєстр дозволених запитів: db.py виконує тільки те, що є в цьому словнику.
ALLOWED_QUERIES = {
    "course_metrics": COURSE_METRICS_SQL,
}
