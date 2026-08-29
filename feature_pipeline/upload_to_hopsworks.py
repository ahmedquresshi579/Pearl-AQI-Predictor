import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "historical_features.csv"
FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY not set — add it to your .env file")

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"{len(df)} rows loaded")

    # Hopsworks requires a primary key and an event-time column.
    # timestamp (unix seconds, unique per row) works as both purposes here.
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project="lahore_aqi_ahmedanjum")
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly Lahore AQI features: pollutants, weather-adjacent, time-based",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    print("Inserting data (this can take a few minutes for ~49k rows)...")
    fg.insert(df)
    print("Done. Data is in the Hopsworks feature store.")


if __name__ == "__main__":
    main()