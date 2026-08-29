"""
Registers the locally trained best_model.pkl into the Hopsworks
Model Registry, along with its evaluation metrics.

Run this AFTER train_model.py has produced best_model.pkl.

Usage:
    python register_model.py
"""
import os
import pickle
import shutil

import hopsworks
from dotenv import load_dotenv

load_dotenv()

MODEL_LOCAL_PATH = "best_model.pkl"
MODEL_DIR = "model_export"  # Hopsworks expects a directory to upload
MODEL_NAME = "lahore_aqi_forecast_model"

# These should match what train_model.py printed — update if you rerun
# training with different results. Kept explicit here rather than
# re-computed, since the registry needs metrics attached, not just
# the model artifact.
METRICS = {
    "rmse": 24.98,
    "mae": 14.64,
    "r2": 0.932,
}


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set — check your .env file")

    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(
            f"{MODEL_LOCAL_PATH} not found — run train_model.py first to produce it"
        )

    with open(MODEL_LOCAL_PATH, "rb") as f:
        model_obj = pickle.load(f)
    print(f"Loaded local model: {type(model_obj).__name__}")

    # Hopsworks model registry wants a directory, not a bare file —
    # copy the pickle into a fresh folder matching its expected layout
    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copy(MODEL_LOCAL_PATH, os.path.join(MODEL_DIR, "model.pkl"))

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    mr = project.get_model_registry()

    print("Registering model...")
    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=METRICS,
        description="Random Forest — Lahore AQI forecasting, trained on OpenWeather historical features",
    )
    model.save(MODEL_DIR)

    print(f"\nModel registered successfully: {MODEL_NAME}, version {model.version}")
    print(f"View it at: https://eu-west.cloud.hopsworks.ai/p/42158/models")


if __name__ == "__main__":
    main()
