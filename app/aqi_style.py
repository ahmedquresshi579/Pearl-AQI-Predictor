"""Shared AQI styling and category helpers for the Streamlit dashboard."""

EPA_CATEGORIES = [
    (0, 50, "Good", "#16C79A"),
    (51, 100, "Moderate", "#F4C95D"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF9B54"),
    (151, 200, "Unhealthy", "#FF5A5F"),
    (201, 300, "Very Unhealthy", "#A66CFF"),
    (301, 500, "Hazardous", "#D93654"),
]


def get_category(aqi: float):
    """Return (label, color_hex) for a given AQI value."""
    for low, high, label, color in EPA_CATEGORIES:
        if low <= aqi <= high:
            return label, color
    return "Hazardous", EPA_CATEGORIES[-1][3]


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(900px 520px at 88% -5%, rgba(37, 211, 190, .10), transparent 60%),
        radial-gradient(720px 500px at 0% 12%, rgba(92, 108, 255, .09), transparent 62%),
        #070B14;
    color: #ECF5F6;
}

.main .block-container {
    max-width: 1450px;
    padding: 2.5rem 3rem 4rem;
}

/* Streamlit chrome */
[data-testid="stHeader"] {
    background: rgba(7, 11, 20, .72);
}

[data-testid="stToolbar"] {
    opacity: .65;
}

[data-testid="stDecoration"] {
    display: none;
}

/* General text */
h1, h2, h3, p, label {
    color: #ECF5F6;
}

[data-testid="stCaptionContainer"] {
    color: #718993;
}

/* Modern section headings */
.section-title,
.modern-section {
    color: #EAF4F5 !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: -.015em !important;
    margin: 2rem 0 .75rem !important;
    padding: 0 0 .65rem .85rem !important;
    border-left: 4px solid #5CE1CF !important;
    border-bottom: 1px solid #182A38 !important;
}

/* Legacy classes retained so every existing HTML block is styled consistently. */
.brand-row {
    margin-bottom: 1rem;
}

.main-header {
    color: #F2F7F8 !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -.045em !important;
    margin: 0 !important;
}

.byline {
    color: #8198A2 !important;
    font-size: .86rem !important;
    margin: .35rem 0 0 !important;
}

.sub-header {
    color: #8EA5AE !important;
    font-size: .93rem !important;
    line-height: 1.55 !important;
    margin: 0 0 1.25rem !important;
}

/* Existing metric-card fallback */
.metric-card {
    min-height: 158px;
    padding: 1.25rem 1.35rem;
    border-radius: 18px;
    border: 1px solid #1A2A39;
    background: linear-gradient(145deg, #101A29, #0B121D);
    box-shadow: 0 14px 34px rgba(0,0,0,.18);
}

.metric-label {
    color: #8299A3;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .12em;
}

.metric-value {
    font-size: 2.45rem;
    line-height: 1;
    font-weight: 800;
    margin: .6rem 0 .75rem;
}

.aqi-badge {
    display: inline-block;
    padding: .32rem .7rem;
    border-radius: 999px;
    font-size: .7rem;
    font-weight: 800;
}

/* Existing alert fallback */
.alert-box {
    border-radius: 14px;
    padding: .9rem 1.1rem;
    font-size: .88rem;
    font-weight: 650;
    margin: 1rem 0;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #192B39;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 16px 35px rgba(0,0,0,.16);
}

/* Charts */
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #203443;
    background: #0D1824;
    color: #DCECEE;
}

.stButton > button:hover {
    border-color: #3AAE9F;
    color: #FFFFFF;
}

/* Responsive */
@media (max-width: 900px) {
    .main .block-container {
        padding: 1.4rem 1rem 3rem;
    }

    .main-header {
        font-size: 1.85rem !important;
    }
}
</style>
"""
