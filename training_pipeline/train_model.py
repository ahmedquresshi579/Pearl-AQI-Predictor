"""
Subsystem 2: Training.

Pulls historical features from the Hopsworks feature store, builds
lag/rolling-average features and multi-horizon DELTA targets (change
in AQI at 24h/48h/72h, not the raw level), trains models to predict
that change, and anchors predictions to the current reading:

    forecast = current_aqi + blend_weight * predicted_delta

A persistence baseline (blend_weight=0) is evaluated alongside every
model, because a forecast that doesn't beat "assume no change" isn't
actually a working forecast — see the reference project that motivated
this rewrite.

Usage:
    python train_model.py
"""
import os
import pickle

import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_OUTPUT_PATH = "best_model.pkl"

BASE_FEATURE_COLUMNS = [
    "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    "hour", "day_of_week",
    "hour_sin", "hour_cos",
    "aqi",  # current reading — the anchor point, and a strong predictor itself
]
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 24, 72]

HORIZONS = {"target_h24": 24, "target_h48": 48, "target_h72": 72}
BLEND_WEIGHT = 0.5  # fixed, not tuned — see note in evaluate()


def load_data_from_hopsworks(api_key: str) -> pd.DataFrame:
    project = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Reading data from feature store (this can take a minute)...")
    df = fg.read()
    print(f"Loaded {len(df)} rows from Hopsworks")
    return df


def build_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds AQI lag features and rolling mean/std on a CONTINUOUS hourly
    time grid, then maps them back onto the actual rows by timestamp —
    not by row position. With ~2% of hours missing, a positional
    .shift(N) would silently pair a row with the wrong past value across
    a gap. Reindexing onto a full hourly DatetimeIndex first, and only
    then shifting/rolling, keeps every lookup anchored to real elapsed
    time.
    """
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    dt_index = pd.to_datetime(df["timestamp"], unit="s")

    aqi_series = pd.Series(df["aqi"].values, index=dt_index)
    full_range = pd.date_range(dt_index.min(), dt_index.max(), freq="h")
    aqi_continuous = aqi_series.reindex(full_range)

    feature_frame = pd.DataFrame(index=full_range)

    for lag_h in LAG_HOURS:
        feature_frame[f"aqi_lag_{lag_h}"] = aqi_continuous.shift(lag_h)

    # shift(1) first so a row's rolling stats never include its own value
    shifted = aqi_continuous.shift(1)
    for window in ROLLING_WINDOWS:
        min_periods = max(1, window // 2)
        feature_frame[f"aqi_rmean_{window}"] = shifted.rolling(window, min_periods=min_periods).mean()
        feature_frame[f"aqi_rstd_{window}"] = shifted.rolling(window, min_periods=min_periods).std()

    df = df.set_index(dt_index)
    df = df.join(feature_frame, how="left")
    df = df.reset_index(drop=True)
    return df


def build_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attaches target_hN (future AQI LEVEL) via exact timestamp lookup,
    then converts each to a DELTA target (future - current) for training."""
    aqi_lookup = df.set_index("timestamp")["aqi"]

    for col_name, hours_ahead in HORIZONS.items():
        offset_seconds = hours_ahead * 3600
        df[col_name] = (df["timestamp"] + offset_seconds).map(aqi_lookup)
        df[f"delta_{col_name}"] = df[col_name] - df["aqi"]

    before = len(df)
    required_cols = list(HORIZONS.keys())
    df = df.dropna(subset=required_cols)
    print(f"Dropped {before - len(df)} rows missing a future target "
          f"(expected near the end of the dataset, and around gaps)")
    return df


def prepare_data(df: pd.DataFrame):
    feature_cols = BASE_FEATURE_COLUMNS + \
        [f"aqi_lag_{h}" for h in LAG_HOURS] + \
        [f"aqi_rmean_{w}" for w in ROLLING_WINDOWS] + \
        [f"aqi_rstd_{w}" for w in ROLLING_WINDOWS]

    delta_cols = [f"delta_{h}" for h in HORIZONS.keys()]
    level_cols = list(HORIZONS.keys())

    df = df.dropna(subset=feature_cols + delta_cols)
    df = df.sort_values("timestamp")

    X = df[feature_cols]
    y_delta = df[delta_cols]
    y_level = df[level_cols]  # kept for evaluation against true future AQI
    current_aqi = df["aqi"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_delta_train, y_delta_test = y_delta.iloc[:split_idx], y_delta.iloc[split_idx:]
    y_level_test = y_level.iloc[split_idx:]
    current_aqi_test = current_aqi.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (chronological split, not random)")
    return X_train, X_test, y_delta_train, y_delta_test, y_level_test, current_aqi_test


def evaluate(name, model, X_test, y_level_test, current_aqi_test):
    """
    Evaluates three things per horizon, all against the TRUE future AQI
    level: the model's anchored blend, the model alone (weight=1), and
    persistence (weight=0). A forecast that doesn't beat persistence is
    not doing useful work, regardless of how good its raw R2 looks.
    """
    predicted_delta = model.predict(X_test)
    print(f"\n{name}")

    for i, col in enumerate(HORIZONS.keys()):
        true_level = y_level_test[col].values
        current = current_aqi_test.values
        delta_pred = predicted_delta[:, i]

        persistence_pred = current  # weight = 0
        model_alone_pred = current + delta_pred  # weight = 1
        blend_pred = current + BLEND_WEIGHT * delta_pred  # weight = 0.5

        def metrics(pred):
            rmse = np.sqrt(mean_squared_error(true_level, pred))
            mae = mean_absolute_error(true_level, pred)
            r2 = r2_score(true_level, pred)
            return rmse, mae, r2

        p_rmse, p_mae, p_r2 = metrics(persistence_pred)
        m_rmse, m_mae, m_r2 = metrics(model_alone_pred)
        b_rmse, b_mae, b_r2 = metrics(blend_pred)

        print(f"  {col}:")
        print(f"    persistence   R2={p_r2:.3f}  RMSE={p_rmse:.2f}  MAE={p_mae:.2f}")
        print(f"    model alone   R2={m_r2:.3f}  RMSE={m_rmse:.2f}  MAE={m_mae:.2f}")
        print(f"    blend (w={BLEND_WEIGHT}) R2={b_r2:.3f}  RMSE={b_rmse:.2f}  MAE={b_mae:.2f}"
              f"  {'[beats persistence]' if b_rmse < p_rmse else '[does NOT beat persistence]'}")

    avg_blend_rmse = np.mean([
        np.sqrt(mean_squared_error(
            y_level_test[col].values,
            current_aqi_test.values + BLEND_WEIGHT * predicted_delta[:, i]
        ))
        for i, col in enumerate(HORIZONS.keys())
    ])
    return {"name": name, "model": model, "avg_blend_rmse": avg_blend_rmse}


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set — check your .env file")

    df = load_data_from_hopsworks(api_key)
    df = build_lag_and_rolling_features(df)
    df = build_forecast_targets(df)

    X_train, X_test, y_delta_train, y_delta_test, y_level_test, current_aqi_test = prepare_data(df)

    results = []

    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    )
    rf.fit(X_train, y_delta_train)
    results.append(evaluate("Random Forest (delta + anchored blend)", rf, X_test, y_level_test, current_aqi_test))

    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train, y_delta_train)
    results.append(evaluate("Ridge Regression (delta + anchored blend)", ridge, X_test, y_level_test, current_aqi_test))

    best = min(results, key=lambda r: r["avg_blend_rmse"])
    print(f"\nBest model: {best['name']} (avg blended RMSE across horizons: {best['avg_blend_rmse']:.2f})")

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(best["model"], f)

    size_mb = os.path.getsize(MODEL_OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved best model to {MODEL_OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"NOTE: this model predicts DELTA, not AQI level. At inference: "
          f"forecast = current_aqi + {BLEND_WEIGHT} * model.predict(features)")


if __name__ == "__main__":
    main()