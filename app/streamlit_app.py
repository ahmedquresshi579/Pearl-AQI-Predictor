"""
Lahore AQI Forecast Dashboard — 10Pearls AQI Predictor.

Pulls recent history from Hopsworks, loads the registered forecasting
model, computes a 24h/48h/72h forecast (delta model, anchored to the
current reading), and shows it alongside historical trends, EDA, a
live AQI gauge, model performance metrics, and SHAP explainability.

IMPORTANT: the feature-building logic here (lag features, rolling
stats) is duplicated from training_pipeline/train_model.py rather than
imported, because the two live in separate deployable folders. If you
change one, change the other — a mismatch here is exactly the kind of
bug that causes a dashboard to silently serve wrong predictions.

Run:
    streamlit run streamlit_app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import hopsworks
from dotenv import load_dotenv

from aqi_style import CUSTOM_CSS, get_category, EPA_CATEGORIES


load_dotenv()


def _html(s: str) -> str:
    """Strip leading whitespace from every line so Streamlit's markdown
    renderer doesn't mistake indented HTML for a code block."""
    return "\n".join(line.strip() for line in s.strip().split("\n"))


FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "lahore_aqi_forecast_model"
BLEND_WEIGHT = 0.5

AUTHOR_NAME = "Muhammad Ahmed Anjum"
PROJECT_TITLE = "10Pearls AQI Predictor"

BASE_FEATURE_COLUMNS = [
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3",
    "hour",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "aqi",
]

LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 24, 72]

HORIZON_LABELS = {
    "target_h24": "24h",
    "target_h48": "48h",
    "target_h72": "72h",
}


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=3600)
def get_hopsworks_project():
    api_key = os.environ.get("HOPSWORKS_API_KEY")

    if not api_key:
        st.error("HOPSWORKS_API_KEY not set. Add it to your .env file.")
        st.stop()

    return hopsworks.login(
        api_key_value=api_key,
        project="lahore_aqi_ahmedanjum"
    )


@st.cache_data(ttl=1800)
def load_recent_data(_project, days_back: int = 120) -> pd.DataFrame:
    fs = _project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df = fg.read()

    df = df.sort_values("timestamp")

    cutoff = df["timestamp"].max() - days_back * 86400

    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


@st.cache_resource(ttl=3600)
def load_model(_project):
    mr = _project.get_model_registry()

    # get_model(version=None) silently defaults to version 1, NOT the
    # latest — that was the real cause of stale metrics/model showing.
    # Fetch all versions and pick the highest one explicitly.
    all_versions = mr.get_models(name=MODEL_NAME)
    model_meta = max(all_versions, key=lambda m: m.version)

    model_dir = model_meta.download()

    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model_obj = pickle.load(f)

    metrics = getattr(model_meta, "training_metrics", None) or {}

    return model_obj, model_meta.version, metrics


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    # Round to the hour so off-schedule fetches (05:01:45, 14:28:37, etc.)
    # land on the same grid as everything else, instead of silently
    # falling off and getting dropped by dropna() downstream.
    hour_bucket = pd.to_datetime(df["timestamp"], unit="s").dt.floor("h")

    # Two readings can floor into the same hour (e.g. a manual trigger
    # plus the scheduled one) — keep the freshest per hour.
    df = df.assign(_hour=hour_bucket)
    df = df.sort_values("timestamp").drop_duplicates(
        subset="_hour",
        keep="last"
    )

    dt_index = df["_hour"]

    aqi_series = pd.Series(
        df["aqi"].values,
        index=dt_index
    )

    full_range = pd.date_range(
        dt_index.min(),
        dt_index.max(),
        freq="h"
    )

    aqi_continuous = aqi_series.reindex(full_range)

    feature_frame = pd.DataFrame(index=full_range)

    for lag_h in LAG_HOURS:
        feature_frame[f"aqi_lag_{lag_h}"] = (
            aqi_continuous.shift(lag_h)
        )

    shifted = aqi_continuous.shift(1)

    for window in ROLLING_WINDOWS:
        min_periods = max(1, window // 2)

        feature_frame[f"aqi_rmean_{window}"] = (
            shifted.rolling(
                window,
                min_periods=min_periods
            ).mean()
        )

        feature_frame[f"aqi_rstd_{window}"] = (
            shifted.rolling(
                window,
                min_periods=min_periods
            ).std()
        )

    df = (
        df
        .set_index(dt_index)
        .join(feature_frame, how="left")
        .reset_index(drop=True)
    )

    df = df.drop(
        columns="_hour",
        errors="ignore"
    )

    return df


def get_feature_columns():

    return (
        BASE_FEATURE_COLUMNS
        + [f"aqi_lag_{h}" for h in LAG_HOURS]
        + [f"aqi_rmean_{w}" for w in ROLLING_WINDOWS]
        + [f"aqi_rstd_{w}" for w in ROLLING_WINDOWS]
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast_latest(
    df: pd.DataFrame,
    model
):

    feature_cols = get_feature_columns()

    latest = (
        df
        .dropna(subset=feature_cols)
        .iloc[[-1]]
    )

    if latest.empty:
        return None

    X = latest[feature_cols]

    current_aqi = latest["aqi"].values[0]

    delta_pred = model.predict(X)[0]

    forecasts = {}

    for i, key in enumerate(
        HORIZON_LABELS.keys()
    ):

        forecasts[key] = (
            current_aqi
            + BLEND_WEIGHT * delta_pred[i]
        )

    return (
        current_aqi,
        forecasts,
        latest["timestamp"].values[0]
    )


# ---------------------------------------------------------------------------
# AQI Gauge
# ---------------------------------------------------------------------------

def make_aqi_gauge(
    value: float,
    title: str
):

    """Create a horizontal bullet-style AQI gauge."""

    label, color = get_category(value)

    steps = [
        {
            "range": [low, high],
            "color": cat_color
        }
        for low, high, _, cat_color
        in EPA_CATEGORIES
    ]

    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=value,

            number={
                "font": {
                    "size": 34,
                    "color": color
                }
            },

            gauge={
                "shape": "bullet",

                "axis": {
                    "range": [0, 500],
                    "tickcolor": "#8FA6A4"
                },

                "bar": {
                    "color": color,
                    "thickness": 0.5
                },

                "steps": steps,

                "bgcolor": "rgba(0,0,0,0)",
            },

            title={
                "text": title,
                "font": {
                    "size": 13,
                    "color": "#8FA6A4"
                }
            },

            domain={
                "x": [0.05, 1],
                "y": [0.2, 0.8]
            },
        )
    )

    fig.update_layout(
        height=90,

        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#E5F2F1"
        },
    )

    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():

    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon="🌫️",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        /* ===================== FINAL DASHBOARD DESIGN ===================== */
        .stApp {
            background:
                radial-gradient(850px 500px at 100% 0%, rgba(40, 196, 178, .11), transparent 58%),
                radial-gradient(700px 500px at 0% 10%, rgba(55, 92, 180, .10), transparent 62%),
                #071018 !important;
            color: #EDF7F7 !important;
        }

        .main .block-container {
            max-width: 1420px !important;
            padding: 2.5rem 3rem 4rem !important;
        }

        /* HERO */
        .dash-hero {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 2rem;
            padding: 0 0 1.8rem;
            border-bottom: 1px solid #19303A;
            margin-bottom: 1.35rem;
        }

        .dash-company {
            color: #5CE1CF;
            font-size: .68rem;
            font-weight: 800;
            letter-spacing: .18em;
            margin-bottom: .55rem;
        }

        .dash-title {
            color: #F3F9F9;
            font-size: 2.7rem;
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -.055em;
            margin: 0;
        }

        .dash-subtitle {
            color: #8198A2;
            font-size: .92rem;
            line-height: 1.55;
            margin-top: .7rem;
            max-width: 760px;
        }

        .dash-author {
            color: #A9BCC1;
            font-size: .78rem;
            margin-top: .8rem;
        }

        .dash-author strong {
            color: #E6F0F1;
            font-weight: 700;
        }

        .dash-status {
            color: #8299A2;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .13em;
            white-space: nowrap;
            padding-bottom: .35rem;
        }

        .dash-status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #35D0BA;
            box-shadow: 0 0 13px rgba(53,208,186,.8);
            margin-right: .42rem;
        }

        /* FORECAST CARDS */
        .forecast-grid {
            display: grid;
            grid-template-columns: 1.08fr repeat(3, 1fr);
            gap: .85rem;
            margin: .75rem 0 .8rem;
        }

        .forecast-tile {
            position: relative;
            min-height: 172px;
            padding: 1.25rem 1.25rem 1.1rem;
            border-radius: 18px;
            border: 1px solid #1A303B;
            background: linear-gradient(145deg, #0F1C27 0%, #0A141D 100%);
            box-shadow: 0 14px 32px rgba(0,0,0,.20);
            overflow: hidden;
            transition: transform .15s ease, border-color .15s ease;
        }

        .forecast-tile:hover {
            transform: translateY(-2px);
            border-color: #2B4A56;
        }

        .forecast-tile.current {
            background: linear-gradient(145deg, #10272A 0%, #0B171B 100%);
        }

        .forecast-tile::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--aqi-color);
        }

        .forecast-head,
        .forecast-foot {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .5rem;
        }

        .forecast-head {
            margin-bottom: 1rem;
        }

        .forecast-label {
            color: #8CA3AC;
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .forecast-time {
            color: #5D747D;
            font-size: .62rem;
            font-weight: 700;
        }

        .forecast-number {
            color: #F5FAFA;
            font-size: 2.55rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: -.055em;
        }

        .forecast-number small {
            color: #718891;
            font-size: .62rem;
            font-weight: 750;
            letter-spacing: .05em;
            margin-left: .25rem;
        }

        .forecast-foot {
            margin-top: 1rem;
        }

        .forecast-status {
            color: var(--aqi-color);
            background: color-mix(in srgb, var(--aqi-color) 13%, transparent);
            border: 1px solid color-mix(in srgb, var(--aqi-color) 30%, transparent);
            border-radius: 999px;
            padding: .3rem .58rem;
            font-size: .63rem;
            font-weight: 800;
            line-height: 1.25;
            max-width: 78%;
        }

        .forecast-delta {
            color: #708791;
            font-size: .62rem;
            font-weight: 750;
            text-align: right;
        }

        .updated-line {
            color: #607780;
            font-size: .67rem;
            margin: .15rem 0 1.2rem;
        }

        /* SECTION HEADINGS */
        .modern-section {
            color: #EAF4F5 !important;
            font-size: 1.02rem !important;
            font-weight: 800 !important;
            letter-spacing: -.01em !important;
            margin: 2rem 0 .8rem !important;
            padding: 0 0 .62rem .8rem !important;
            border-left: 4px solid #5CE1CF !important;
            border-bottom: 1px solid #19303A !important;
        }

        /* MODEL METRICS */
        .metrics-strip {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            gap: .75rem !important;
            margin: .5rem 0 .8rem !important;
        }

        .metric-pill {
            padding: .9rem 1rem !important;
            border-radius: 14px !important;
            background: #0B1721 !important;
            border: 1px solid #19303A !important;
        }

        .metric-pill-label {
            color: #708791 !important;
            font-size: .63rem !important;
            font-weight: 800 !important;
            letter-spacing: .09em !important;
        }

        .metric-pill-value {
            color: #5CE1CF !important;
            font-size: 1.18rem !important;
            font-weight: 850 !important;
            margin-top: .25rem !important;
        }

        /* TABLES + CHARTS */
        [data-testid="stDataFrame"] {
            border: 1px solid #19303A !important;
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 12px 30px rgba(0,0,0,.14) !important;
        }

        [data-testid="stPlotlyChart"] {
            border-radius: 15px !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #657C85 !important;
        }

        /* ALERT */
        .alert-box {
            border-radius: 13px !important;
            padding: .85rem 1rem !important;
            margin: .8rem 0 1rem !important;
            font-size: .84rem !important;
            font-weight: 650 !important;
        }

        /* RESPONSIVE */
        @media (max-width: 950px) {
            .main .block-container {
                padding: 1.4rem 1rem 3rem !important;
            }

            .dash-hero {
                display: block;
            }

            .dash-status {
                margin-top: 1rem;
            }

            .forecast-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .metrics-strip {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }

        @media (max-width: 600px) {
            .forecast-grid,
            .metrics-strip {
                grid-template-columns: 1fr !important;
            }

            .dash-title {
                font-size: 2.05rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------------
    # Dashboard hero
    # -----------------------------------------------------------------------

    st.markdown(
        _html(f"""
        <div class="dash-hero">
            <div>
                <div class="dash-company">10PEARLS · DATA SCIENCE INTERNSHIP PROGRAMME</div>
                <div class="dash-title">Lahore Air Quality</div>
                <div class="dash-subtitle">
                    A live 72-hour AQI forecast combining recent observations,
                    historical patterns, and the registered forecasting model.
                </div>
                <div class="dash-author">
                    Built by <strong>{AUTHOR_NAME}</strong> · {PROJECT_TITLE}
                </div>
            </div>
            <div class="dash-status">
                <span class="dash-status-dot"></span>
                FORECAST ONLINE
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Load Hopsworks + data + model
    # -----------------------------------------------------------------------

    with st.spinner(
        "Connecting to Hopsworks..."
    ):

        project = get_hopsworks_project()

        raw_df = load_recent_data(
            project
        )

        model, model_version, model_metrics = load_model(
            project
        )

    # -----------------------------------------------------------------------
    # Build features
    # -----------------------------------------------------------------------

    df = build_features(
        raw_df
    )

    result = forecast_latest(
        df,
        model
    )

    if result is None:

        st.error(
            "Not enough recent history to compute "
            "lag/rolling features yet."
        )

        st.stop()

    current_aqi, forecasts, latest_ts = result

    latest_dt = pd.to_datetime(
        latest_ts,
        unit="s"
    )

    # -----------------------------------------------------------------------
    # Current conditions + forecasts
    # -----------------------------------------------------------------------

    current_label, current_color = get_category(current_aqi)

    forecast_items = [
        (
            "Current AQI",
            "NOW",
            current_aqi,
            current_label,
            current_color,
            0
        ),
    ]

    for key, hlabel in HORIZON_LABELS.items():
        val = forecasts[key]
        flabel, fcolor = get_category(val)

        forecast_items.append(
            (
                hlabel,
                "OUTLOOK",
                val,
                flabel,
                fcolor,
                val - current_aqi
            )
        )

    cards_html = '<div class="forecast-grid">'

    for idx, (
        title,
        time_label,
        value,
        status,
        tile_color,
        delta
    ) in enumerate(forecast_items):

        if idx == 0:
            delta_text = "Latest observation"
            tile_class = "forecast-tile current"

        else:
            sign = "+" if delta >= 0 else ""
            delta_text = f"{sign}{delta:.0f} vs now"
            tile_class = "forecast-tile"

        cards_html += f"""
        <div class="{tile_class}" style="--aqi-color:{tile_color};">
            <div class="forecast-head">
                <span class="forecast-label">{title}</span>
                <span class="forecast-time">{time_label}</span>
            </div>

            <div class="forecast-number">
                {value:.0f}<small>AQI</small>
            </div>

            <div class="forecast-foot">
                <span class="forecast-status">{status}</span>
                <span class="forecast-delta">{delta_text}</span>
            </div>
        </div>
        """

    cards_html += "</div>"

    st.markdown(
        _html(cards_html),
        unsafe_allow_html=True,
    )

    st.markdown(
        _html(
            f'<div class="updated-line">'
            f'Updated {latest_dt.strftime("%d %B %Y, %H:%M UTC")} · '
            f'Forecast model v{model_version}'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )

    # ADDED: UTC / PKT clarification
    st.caption(
        "All times shown are UTC. Lahore local time (PKT) = UTC + 5 hours."
    )

    # -----------------------------------------------------------------------
    # Hazard alert
    # -----------------------------------------------------------------------

    max_forecast = max(
        forecasts.values()
    )

    if max_forecast >= 150:

        worst_label, worst_color = get_category(
            max_forecast
        )

        st.markdown(
            _html(f"""
            <div class="alert-box" style="background:{worst_color}22;border:1px solid {worst_color};color:{worst_color};">
                ⚠️ {worst_label} air quality expected within 3 days (AQI {max_forecast:.0f}).
                Sensitive groups should limit prolonged outdoor exertion.
            </div>
            """),
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # AQI outlook
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">AQI Outlook</p>',
        unsafe_allow_html=True
    )

    outlook_labels = [
        "Now",
        "24h",
        "48h",
        "72h"
    ]

    outlook_values = (
        [current_aqi]
        + [
            forecasts[k]
            for k in HORIZON_LABELS
        ]
    )

    outlook_colors = [
        get_category(v)[1]
        for v in outlook_values
    ]

    outlook_fig = go.Figure(
        go.Bar(
            x=outlook_values,
            y=outlook_labels,
            orientation="h",
            marker=dict(
                color=outlook_colors
            ),
            text=[
                f"{v:.0f}"
                for v in outlook_values
            ],
            textposition="outside",
            hovertemplate=(
                "%{y}: %{x:.0f} AQI"
                "<extra></extra>"
            ),
        )
    )

    outlook_fig.update_layout(
        template="plotly_dark",
        height=225,
        margin=dict(
            l=10,
            r=55,
            t=8,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#B7CDD1"
        ),
        xaxis=dict(
            title=None,
            range=[
                0,
                max(
                    170,
                    max(outlook_values) * 1.18
                )
            ],
            gridcolor="rgba(130,160,170,.10)",
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            autorange="reversed"
        ),
        showlegend=False,
    )

    st.plotly_chart(
        outlook_fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

    # -----------------------------------------------------------------------
    # Model performance
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">Model Performance</p>',
        unsafe_allow_html=True
    )

    if model_metrics:

        pill_html = (
            '<div class="metrics-strip">'
        )

        for key, val in model_metrics.items():

            display_val = (
                f"{val:.3f}"
                if isinstance(
                    val,
                    (int, float)
                )
                else str(val)
            )

            pill_html += (
                '<div class="metric-pill">'

                f'<div class="metric-pill-label">'
                f'{key}'
                f'</div>'

                f'<div class="metric-pill-value">'
                f'{display_val}'
                f'</div>'

                '</div>'
            )

        pill_html += "</div>"

        st.markdown(
            pill_html,
            unsafe_allow_html=True
        )

        st.caption(
            "Metrics computed on a held-out "
            "chronological test split, logged at "
            "model registration time."
        )

    else:

        st.info(
            "No training metrics were found on "
            "this model version in the registry."
        )

    # -----------------------------------------------------------------------
    # Historical trend
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">Historical AQI Trend</p>',
        unsafe_allow_html=True
    )

    trend_df = (
        df
        .dropna(subset=["aqi"])
        .copy()
    )

    trend_df["datetime"] = pd.to_datetime(
        trend_df["timestamp"],
        unit="s"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df["datetime"],
            y=trend_df["aqi"],
            mode="lines",

            line={
                "color": "#4ECDC4",
                "width": 1.5
            },

            name="AQI",
        )
    )

    for low, high, cat_label, cat_color in EPA_CATEGORIES:

        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=cat_color,
            opacity=0.06,
            line_width=0
        )

    fig.update_layout(
        template="plotly_dark",

        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",

        height=380,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        xaxis_title=None,
        yaxis_title="AQI",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------------------------
    # Exploratory analysis
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">Exploratory Analysis</p>',
        unsafe_allow_html=True
    )

    eda_cols = st.columns(2)

    with eda_cols[0]:

        hourly_mean = (
            trend_df
            .groupby(
                trend_df["datetime"].dt.hour
            )["aqi"]
            .mean()
            .reset_index()
        )

        hourly_mean.columns = [
            "hour",
            "mean_aqi"
        ]

        fig_hour = px.bar(
            hourly_mean,
            x="hour",
            y="mean_aqi",
            title="Mean AQI by Hour of Day (UTC)",
            color="mean_aqi",
            color_continuous_scale=[
                "#2ECC71",
                "#F4D35E",
                "#EF476F"
            ]
        )

        fig_hour.update_layout(
            template="plotly_dark",

            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            height=340,

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig_hour,
            use_container_width=True
        )

    with eda_cols[1]:

        def categorize(v):
            return get_category(v)[0]

        trend_df["category"] = (
            trend_df["aqi"]
            .apply(categorize)
        )

        cat_counts = (
            trend_df["category"]
            .value_counts()
            .reset_index()
        )

        cat_counts.columns = [
            "category",
            "hours"
        ]

        color_map = {
            c[2]: c[3]
            for c in EPA_CATEGORIES
        }

        fig_cat = px.pie(
            cat_counts,
            names="category",
            values="hours",
            hole=0.55,
            title="Share of Time in Each EPA Category",
            color="category",
            color_discrete_map=color_map
        )

        fig_cat.update_layout(
            template="plotly_dark",

            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            height=340
        )

        st.plotly_chart(
            fig_cat,
            use_container_width=True
        )

    # -----------------------------------------------------------------------
    # SHAP explainability
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">72h Forecast Drivers</p>',
        unsafe_allow_html=True
    )

    try:

        import shap

        horizon_idx = list(
            HORIZON_LABELS.keys()
        ).index(
            "target_h72"
        )

        sub_model = model.estimators_[
            horizon_idx
        ]

        feature_cols = get_feature_columns()

        sample_df = (
            df
            .dropna(
                subset=feature_cols
            )
            .tail(200)[feature_cols]
        )

        explainer = shap.Explainer(
            sub_model,
            sample_df
        )

        shap_values = explainer(
            sample_df
        )

        mean_abs_shap = (
            np.abs(
                shap_values.values
            ).mean(axis=0)
        )

        shap_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": mean_abs_shap
            }
        )

        shap_df = (
            shap_df
            .sort_values(
                "mean_abs_shap",
                ascending=True
            )
            .tail(15)
        )

        fig_shap = px.bar(
            shap_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title=(
                "Mean |SHAP value| — average impact "
                "on the predicted 72h AQI change"
            )
        )

        fig_shap.update_traces(
            marker_color="#4ECDC4"
        )

        fig_shap.update_layout(
            template="plotly_dark",

            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            height=420,

            xaxis_title="mean(|SHAP value|)",
            yaxis_title=None
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

        # -------------------------------------------------------------------
        # ADDED: SHAP beeswarm showing direction of impact
        # -------------------------------------------------------------------

        st.markdown(
            "**Direction of impact (SHAP beeswarm)**"
        )

        import matplotlib.pyplot as plt

        fig_beeswarm, ax = plt.subplots(
            figsize=(9, 5)
        )

        shap.plots.beeswarm(
            shap_values,
            show=False,
            max_display=15
        )

        plt.gcf().set_facecolor(
            "#0B1721"
        )

        ax.set_facecolor(
            "#0B1721"
        )

        ax.tick_params(
            colors="#B7CDD1"
        )

        for text in (
            ax.get_yticklabels()
            + ax.get_xticklabels()
        ):
            text.set_color(
                "#B7CDD1"
            )

        st.pyplot(
            fig_beeswarm,
            use_container_width=True
        )

        plt.close(
            fig_beeswarm
        )

        st.caption(
            "SHAP values show each feature's average "
            "contribution to the predicted CHANGE in AQI "
            "over 72h, computed on the 200 most recent "
            "complete rows."
        )

    except Exception as e:

        st.info(
            f"Could not compute SHAP values: {e}"
        )

    # -----------------------------------------------------------------------
    # Recent readings
    # -----------------------------------------------------------------------

    st.markdown(
        '<p class="modern-section">Recent Readings</p>',
        unsafe_allow_html=True
    )

    display_cols = [
        "datetime",
        "aqi",
        "pm10",
        "co",
        "no2",
        "o3",
        "so2"
    ]

    recent = (
        trend_df[display_cols]
        .tail(48)
        .sort_values(
            "datetime",
            ascending=False
        )
    )

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------

    st.markdown(
        _html(f"""
        <p style="color:#667F88;font-size:0.76rem;margin-top:2.5rem;">
            {PROJECT_TITLE} · Built by {AUTHOR_NAME} · 10Pearls Data Science Internship Programme
        </p>
        """),
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()