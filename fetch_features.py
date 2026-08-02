"""
Fetches current air pollution + weather data for Lahore from OpenWeather,
converts PM2.5 to US EPA AQI, and computes model-ready features.

Usage:
    python fetch_features.py

Requires:
    OPENWEATHER_API_KEY set as an environment variable (see .env.example)
"""
import os
import math
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from aqi_utils import pm25_to_aqi

load_dotenv()

LAT = 31.5497
LON = 74.3436
CURRENT_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"


def fetch_current_raw(lat: float, lon: float, api_key: str) -> dict:
    """Fetch the current air pollution reading from OpenWeather."""
    resp = requests.get(
        CURRENT_URL,
        params={"lat": lat, "lon": lon, "appid": api_key},
        timeout=15,
    )
    resp.raise_for_status()  # fail loudly on 401/429/500, don't limp forward with bad data
    data = resp.json()

    if not data.get("list"):
        raise ValueError(f"OpenWeather returned no data for lat={lat}, lon={lon}: {data}")

    return data["list"][0]


def fetch_historical_raw(lat: float, lon: float, api_key: str, start: int, end: int) -> list:
    """Fetch historical air pollution readings between two Unix timestamps."""
    resp = requests.get(
        HISTORY_URL,
        params={"lat": lat, "lon": lon, "start": start, "end": end, "appid": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", [])


def compute_features(raw_record: dict) -> dict:
    """
    Turn one raw OpenWeather record into a flat feature dict.
    raw_record is one entry from the 'list' array (current or historical).
    """
    components = raw_record["components"]
    dt_utc = datetime.fromtimestamp(raw_record["dt"], tz=timezone.utc)

    pm25 = components.get("pm2_5")
    if pm25 is None:
        raise ValueError(f"Record missing pm2_5, cannot compute AQI: {raw_record}")

    aqi = pm25_to_aqi(pm25)

    hour = dt_utc.hour
    day_of_week = dt_utc.weekday()  # 0=Monday
    month = dt_utc.month

    return {
        # identifiers / raw target
        "timestamp": raw_record["dt"],
        "datetime_utc": dt_utc.isoformat(),
        "aqi": aqi,

        # raw pollutant features
        "pm2_5": pm25,
        "pm10": components.get("pm10"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "nh3": components.get("nh3"),

        # time-based features
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        # cyclical encoding — tells the model 23:00 and 00:00 are adjacent,
        # not maximally far apart the way raw integers would imply
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


if __name__ == "__main__":
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENWEATHER_API_KEY not set. Create a .env file (see .env.example) "
            "or set it in your shell before running this script."
        )

    print("Fetching current air pollution data for Lahore...")
    raw = fetch_current_raw(LAT, LON, api_key)
    features = compute_features(raw)

    print("\nComputed feature record:")
    for k, v in features.items():
        print(f"  {k:15s}: {v}")