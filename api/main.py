"""
10Pearls AQI Predictor — REST API.

Exposes the same forecast served by the Streamlit dashboard as JSON
endpoints, for programmatic access.

IMPORTANT: feature-building and forecast logic here is duplicated from
app/streamlit_app.py and training_pipeline/train_model.py, not
imported — same tradeoff made throughout this project (separate
deployable services, no shared package). If the lag/rolling feature
logic changes in one place, it must change in all three, or this
service will silently serve predictions computed differently than the
model was trained on.

Run locally:
    uvicorn main:app --reload

Endpoints:
    GET /health    — liveness check, never contacts Hopsworks directly
    GET /current   — latest AQI reading
    GET /forecast  — 24h/48h/72h AQI forecast
"""
import os
import pickle
import time
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "lahore_aqi_forecast_model"
BLEND_WEIGHT = 0.5
CACHE_TTL_SECONDS = 1800  # 30 min, matches the dashboard's cache TTL

BASE_FEATURE_COLUMNS = [
    "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    "hour", "day_of_week", "hour_sin", "hour_cos", "aqi",
]
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 24, 72]
HORIZON_LABELS = {"target_h24": "24h", "target_h48": "48h", "target_h72": "72h"}

EPA_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def get_category(aqi: float) -> str:
    for low, high, label in EPA_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Hazardous"


app = FastAPI(
    title="10Pearls AQI Predictor API",
    description="24h/48h/72h AQI forecast for Lahore, served from the same "
                 "delta-model pipeline as the Streamlit dashboard.",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Cache — models load lazily on first request, not at API startup, so a
# transient Hopsworks outage doesn't stop the service from booting.
# ---------------------------------------------------------------------------
_cache = {
    "project": None,
    "model": None,
    "model_version": None,
    "data": None,
    "data_loaded_at": None,
    "last_successful_forecast": None,  # what /health reports, never a live call
}


def _get_project():
    if _cache["project"] is None:
        api_key = os.environ.get("HOPSWORKS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="HOPSWORKS_API_KEY not configured")
        _cache["project"] = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    return _cache["project"]


def _load_data(force: bool = False) -> pd.DataFrame:
    now = time.time()
    stale = (
        _cache["data"] is None
        or _cache["data_loaded_at"] is None
        or (now - _cache["data_loaded_at"]) > CACHE_TTL_SECONDS
    )
    if force or stale:
        project = _get_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read().sort_values("timestamp")
        cutoff = df["timestamp"].max() - 120 * 86400
        _cache["data"] = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        _cache["data_loaded_at"] = now
    return _cache["data"]


def _load_model():
    if _cache["model"] is None:
        project = _get_project()
        mr = project.get_model_registry()
        all_versions = mr.get_models(name=MODEL_NAME)
        model_meta = max(all_versions, key=lambda m: m.version)
        model_dir = model_meta.download()
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            _cache["model"] = pickle.load(f)
        _cache["model_version"] = model_meta.version
    return _cache["model"], _cache["model_version"]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    hour_bucket = pd.to_datetime(df["timestamp"], unit="s").dt.floor("h")
    df = df.assign(_hour=hour_bucket)
    df = df.sort_values("timestamp").drop_duplicates(subset="_hour", keep="last")
    dt_index = df["_hour"]

    aqi_series = pd.Series(df["aqi"].values, index=dt_index)
    full_range = pd.date_range(dt_index.min(), dt_index.max(), freq="h")
    aqi_continuous = aqi_series.reindex(full_range)

    feature_frame = pd.DataFrame(index=full_range)
    for lag_h in LAG_HOURS:
        feature_frame[f"aqi_lag_{lag_h}"] = aqi_continuous.shift(lag_h)
    shifted = aqi_continuous.shift(1)
    for window in ROLLING_WINDOWS:
        min_periods = max(1, window // 2)
        feature_frame[f"aqi_rmean_{window}"] = shifted.rolling(window, min_periods=min_periods).mean()
        feature_frame[f"aqi_rstd_{window}"] = shifted.rolling(window, min_periods=min_periods).std()

    df = df.set_index(dt_index).join(feature_frame, how="left").reset_index(drop=True)
    return df.drop(columns="_hour", errors="ignore")


def _get_feature_columns():
    return BASE_FEATURE_COLUMNS + \
        [f"aqi_lag_{h}" for h in LAG_HOURS] + \
        [f"aqi_rmean_{w}" for w in ROLLING_WINDOWS] + \
        [f"aqi_rstd_{w}" for w in ROLLING_WINDOWS]


def _compute_forecast():
    raw_df = _load_data()
    df = _build_features(raw_df)
    model, model_version = _load_model()

    feature_cols = _get_feature_columns()
    latest = df.dropna(subset=feature_cols).iloc[[-1]]
    if latest.empty:
        raise HTTPException(status_code=503, detail="Not enough recent history to forecast yet")

    X = latest[feature_cols]
    current_aqi = float(latest["aqi"].values[0])
    delta_pred = model.predict(X)[0]

    forecasts = {}
    for i, key in enumerate(HORIZON_LABELS.keys()):
        forecasts[HORIZON_LABELS[key]] = round(current_aqi + BLEND_WEIGHT * delta_pred[i], 1)

    result = {
        "current_aqi": round(current_aqi, 1),
        "current_category": get_category(current_aqi),
        "observed_at": pd.to_datetime(latest["timestamp"].values[0], unit="s").isoformat() + "Z",
        "forecast": {
            horizon: {"aqi": val, "category": get_category(val)}
            for horizon, val in forecasts.items()
        },
        "model_version": model_version,
        "blend_weight": BLEND_WEIGHT,
    }
    _cache["last_successful_forecast"] = result
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check. Reports cached state only — never triggers a live
    Hopsworks login, so a health probe can't itself become the load that
    takes the service down."""
    return {
        "status": "ok",
        "last_successful_forecast_at": (
            _cache["last_successful_forecast"]["observed_at"]
            if _cache["last_successful_forecast"] else None
        ),
    }


@app.get("/current")
def current():
    """Current AQI reading and category."""
    result = _compute_forecast()
    return {
        "aqi": result["current_aqi"],
        "category": result["current_category"],
        "observed_at": result["observed_at"],
    }


@app.get("/forecast")
def forecast():
    """24h / 48h / 72h AQI forecast."""
    return _compute_forecast()


@app.get("/")
def root():
    return {
        "name": "10Pearls AQI Predictor API",
        "endpoints": ["/health", "/current", "/forecast", "/docs"],
    }