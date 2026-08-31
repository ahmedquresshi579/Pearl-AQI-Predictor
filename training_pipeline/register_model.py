"""
Registers the locally trained best_model.pkl (multi-horizon delta model)
into the Hopsworks Model Registry, with the ACTUAL metrics from the
delta/anchored-blend training run — not the old single-target nowcast
model's numbers, which is what was incorrectly attached previously.
"""
import os
import pickle
import shutil

import hopsworks
from dotenv import load_dotenv

load_dotenv()

MODEL_LOCAL_PATH = "best_model.pkl"
MODEL_DIR = "model_export"
MODEL_NAME = "lahore_aqi_forecast_model"

BLEND_WEIGHT = 0.5

METRICS = {
    "blend_weight": BLEND_WEIGHT,
    "r2_24h": 0.822,
    "r2_48h": 0.737,
    "r2_72h": 0.690,
    "rmse_24h": 40.45,
    "rmse_48h": 49.66,
    "rmse_72h": 54.00,
}

DESCRIPTION = (
    "Multi-horizon (24h/48h/72h) Lahore AQI forecaster. "
    "Predicts the CHANGE in AQI, not the level — at inference, "
    f"forecast = current_aqi + {BLEND_WEIGHT} * model.predict(features). "
    "Uses lag features (1-24h) and rolling mean/std (3h/24h/72h). "
    "Beats a persistence baseline at all three horizons."
)


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set")

    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(f"{MODEL_LOCAL_PATH} not found — run train_model.py first")

    with open(MODEL_LOCAL_PATH, "rb") as f:
        model_obj = pickle.load(f)
    print(f"Loaded local model: {type(model_obj).__name__}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copy(MODEL_LOCAL_PATH, os.path.join(MODEL_DIR, "model.pkl"))

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    mr = project.get_model_registry()

    print("Registering model with correct metrics...")
    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=METRICS,
        description=DESCRIPTION,
    )
    model.save(MODEL_DIR)

    print(f"\nRegistered: {MODEL_NAME}, version {model.version}")


if __name__ == "__main__":
    main()