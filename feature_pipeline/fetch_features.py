"""
Fetches the last 48 hours of air pollution data for Lahore from
OpenWeather (self-healing: covers any gap up to 48h from a missed or
delayed scheduled run), converts PM2.5 to AQI, computes features, and
writes all rows into the Hopsworks feature store in one batch.

Because Hopsworks upserts by primary key (timestamp), re-sending hours
that already exist is safe — they just get skipped/updated, never
duplicated. This makes the hourly job self-healing against GitHub
Actions' scheduling delays, without needing a separate backfill run.

Usage:
    python fetch_features.py
"""
import os
import math
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

from aqi_utils import pm25_to_aqi

load_dotenv()

LAT = 31.5497
LON = 74.3436
HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
LOOKBACK_HOURS = 48

NUMERIC_FLOAT_COLUMNS = [
    "aqi", "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]


def fetch_recent_raw(lat: float, lon: float, api_key: str, hours_back: int) -> list:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)
    resp = requests.get(
        HISTORY_URL,
        params={"lat": lat, "lon": lon, "start": int(start.timestamp()), "end": int(end.timestamp()), "appid": api_key},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", [])


def compute_features(raw_record: dict) -> dict:
    components = raw_record["components"]
    dt_utc = datetime.fromtimestamp(raw_record["dt"], tz=timezone.utc)

    pm25 = components.get("pm2_5")
    if pm25 is None:
        raise ValueError(f"Record missing pm2_5: {raw_record}")

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


def write_batch_to_hopsworks(rows: list, hopsworks_api_key: str):
    project = hopsworks.login(api_key_value=hopsworks_api_key, project="lahore_aqi_ahmedanjum")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df = pd.DataFrame(rows)
    for col in NUMERIC_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    fg.insert(df)
    print(f"Upserted {len(df)} rows into '{FEATURE_GROUP_NAME}' "
          f"(covers {df['datetime_utc'].min()} -> {df['datetime_utc'].max()})")


if __name__ == "__main__":
    openweather_key = os.environ.get("OPENWEATHER_API_KEY")
    hopsworks_key = os.environ.get("HOPSWORKS_API_KEY")

    if not openweather_key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set")
    if not hopsworks_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set")

    print(f"Fetching last {LOOKBACK_HOURS}h of air pollution data for Lahore...")
    raw_records = fetch_recent_raw(LAT, LON, openweather_key, LOOKBACK_HOURS)
    print(f"Got {len(raw_records)} raw records")

    rows = []
    for raw in raw_records:
        try:
            rows.append(compute_features(raw))
        except ValueError as e:
            print(f"Skipping bad record: {e}")
            continue

    if not rows:
        raise RuntimeError("No valid rows computed — nothing to write")

    print("Writing batch to Hopsworks feature store...")
    write_batch_to_hopsworks(rows, hopsworks_key)