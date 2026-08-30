"""
Fetches current air pollution + weather data for Lahore from OpenWeather,
converts PM2.5 to US EPA AQI, computes model-ready features, and writes
the resulting row into the Hopsworks feature store.

BUG FIX: earlier versions of this script computed features and only
printed them — they never actually wrote to Hopsworks, so the hourly
GitHub Action was succeeding while doing nothing useful. This version
inserts the row for real.

Usage:
    python fetch_features.py

Requires:
    OPENWEATHER_API_KEY and HOPSWORKS_API_KEY set as environment variables
"""
import os
import math
from datetime import datetime, timezone

import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

from aqi_utils import pm25_to_aqi

load_dotenv()

LAT = 31.5497
LON = 74.3436
CURRENT_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1


def fetch_current_raw(lat: float, lon: float, api_key: str) -> dict:
    resp = requests.get(CURRENT_URL, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("list"):
        raise ValueError(f"OpenWeather returned no data for lat={lat}, lon={lon}: {data}")
    return data["list"][0]


def fetch_historical_raw(lat: float, lon: float, api_key: str, start: int, end: int) -> list:
    resp = requests.get(HISTORY_URL, params={"lat": lat, "lon": lon, "start": start, "end": end, "appid": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", [])


def compute_features(raw_record: dict) -> dict:
    components = raw_record["components"]
    dt_utc = datetime.fromtimestamp(raw_record["dt"], tz=timezone.utc)

    pm25 = components.get("pm2_5")
    if pm25 is None:
        raise ValueError(f"Record missing pm2_5, cannot compute AQI: {raw_record}")

    aqi = pm25_to_aqi(pm25)
    hour = dt_utc.hour
    day_of_week = dt_utc.weekday()
    month = dt_utc.month

    return {
        "timestamp": raw_record["dt"],
        "datetime_utc": dt_utc.isoformat(),
        "aqi": aqi,
        "pm2_5": pm25,
        "pm10": components.get("pm10"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "nh3": components.get("nh3"),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def write_to_hopsworks(features: dict, hopsworks_api_key: str):
    project = hopsworks.login(api_key_value=hopsworks_api_key, project="lahore_aqi_ahmedanjum")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df = pd.DataFrame([features])
    fg.insert(df)
    print(f"Inserted 1 row into '{FEATURE_GROUP_NAME}' (timestamp={features['timestamp']})")


if __name__ == "__main__":
    openweather_key = os.environ.get("OPENWEATHER_API_KEY")
    hopsworks_key = os.environ.get("HOPSWORKS_API_KEY")

    if not openweather_key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set")
    if not hopsworks_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set — required to write to the feature store")

    print("Fetching current air pollution data for Lahore...")
    raw = fetch_current_raw(LAT, LON, openweather_key)
    features = compute_features(raw)

    print("Computed feature record:")
    for k, v in features.items():
        print(f"  {k:15s}: {v}")

    print("\nWriting to Hopsworks feature store...")
    write_to_hopsworks(features, hopsworks_key)