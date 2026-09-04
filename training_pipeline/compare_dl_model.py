"""
Adds a small TensorFlow/Keras neural network to the model comparison,
alongside Ridge and Random Forest, on the exact same features, target,
split, and delta+anchor evaluation as train_model.py.

This is a STANDALONE comparison script — it does NOT overwrite
best_model.pkl or touch the deployed model. Run it, look at the
results, and decide whether the neural net actually beats Ridge before
changing anything in production.

Usage:
    python compare_dl_model.py
"""
import os

import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

load_dotenv()

FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1

BASE_FEATURE_COLUMNS = [
    "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    "hour", "day_of_week", "hour_sin", "hour_cos", "aqi",
]
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 24, 72]
HORIZONS = {"target_h24": 24, "target_h48": 48, "target_h72": 72}
BLEND_WEIGHT = 0.5


def load_data_from_hopsworks(api_key: str) -> pd.DataFrame:
    project = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    print("Reading data from feature store...")
    df = fg.read()
    print(f"Loaded {len(df)} rows")
    return df


def build_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    dt_index = pd.to_datetime(df["timestamp"], unit="s")
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
    return df


def build_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    aqi_lookup = df.set_index("timestamp")["aqi"]
    for col_name, hours_ahead in HORIZONS.items():
        offset_seconds = hours_ahead * 3600
        df[col_name] = (df["timestamp"] + offset_seconds).map(aqi_lookup)
        df[f"delta_{col_name}"] = df[col_name] - df["aqi"]
    before = len(df)
    df = df.dropna(subset=list(HORIZONS.keys()))
    print(f"Dropped {before - len(df)} rows missing a future target")
    return df


def get_feature_columns():
    return BASE_FEATURE_COLUMNS + \
        [f"aqi_lag_{h}" for h in LAG_HOURS] + \
        [f"aqi_rmean_{w}" for w in ROLLING_WINDOWS] + \
        [f"aqi_rstd_{w}" for w in ROLLING_WINDOWS]


def prepare_data(df: pd.DataFrame):
    feature_cols = get_feature_columns()
    delta_cols = [f"delta_{h}" for h in HORIZONS.keys()]
    level_cols = list(HORIZONS.keys())

    df = df.dropna(subset=feature_cols + delta_cols)
    df = df.sort_values("timestamp")

    X = df[feature_cols]
    y_delta = df[delta_cols]
    y_level = df[level_cols]
    current_aqi = df["aqi"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_delta_train, y_delta_test = y_delta.iloc[:split_idx], y_delta.iloc[split_idx:]
    y_level_test = y_level.iloc[split_idx:]
    current_aqi_test = current_aqi.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_delta_train, y_delta_test, y_level_test, current_aqi_test


def evaluate(name, predict_fn, X_test, y_level_test, current_aqi_test):
    predicted_delta = predict_fn(X_test)
    print(f"\n{name}")
    results = {}
    for i, col in enumerate(HORIZONS.keys()):
        true_level = y_level_test[col].values
        current = current_aqi_test.values
        delta_pred = predicted_delta[:, i]

        blend_pred = current + BLEND_WEIGHT * delta_pred
        rmse = np.sqrt(mean_squared_error(true_level, blend_pred))
        mae = mean_absolute_error(true_level, blend_pred)
        r2 = r2_score(true_level, blend_pred)
        results[col] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {col}: blend R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    avg_rmse = np.mean([v["rmse"] for v in results.values()])
    return {"name": name, "avg_rmse": avg_rmse, "per_horizon": results}


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set")

    df = load_data_from_hopsworks(api_key)
    df = build_lag_and_rolling_features(df)
    df = build_forecast_targets(df)
    X_train, X_test, y_delta_train, y_delta_test, y_level_test, current_aqi_test = prepare_data(df)

    results = []

    # --- Ridge (existing baseline, for direct comparison on this run) ---
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train, y_delta_train)
    results.append(evaluate("Ridge Regression (blend)", ridge.predict, X_test, y_level_test, current_aqi_test))

    # --- Random Forest (existing baseline) ---
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    rf.fit(X_train, y_delta_train)
    results.append(evaluate("Random Forest (blend)", rf.predict, X_test, y_level_test, current_aqi_test))

    # --- Small neural network (new) ---
    print("\nTraining neural network (TensorFlow/Keras)...")
    import tensorflow as tf
    from tensorflow import keras

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tf.random.set_seed(42)
    nn_model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3),  # 3 outputs: delta_h24, delta_h48, delta_h72
    ])
    nn_model.compile(optimizer="adam", loss="mse")
    nn_model.fit(
        X_train_scaled, y_delta_train.values,
        epochs=30, batch_size=64, verbose=0,
        validation_split=0.1,
    )

    def nn_predict(X):
        return nn_model.predict(scaler.transform(X), verbose=0)

    results.append(evaluate("Neural Network (Keras MLP, blend)", nn_predict, X_test, y_level_test, current_aqi_test))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (avg RMSE across 3 horizons, lower is better)")
    for r in sorted(results, key=lambda x: x["avg_rmse"]):
        print(f"  {r['name']:40s} avg RMSE = {r['avg_rmse']:.2f}")


if __name__ == "__main__":
    main()