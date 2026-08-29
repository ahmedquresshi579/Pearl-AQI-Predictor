
import os
import csv
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from fetch_features import fetch_historical_raw, compute_features, LAT, LON

load_dotenv()

BACKFILL_START = datetime(2020, 11, 27, tzinfo=timezone.utc)
CHUNK_DAYS = 30
OUTPUT_FILE = "historical_features.csv"
REQUEST_DELAY_SECONDS = 1


def daterange_chunks(start, end, chunk_days):
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def run_backfill(api_key):
    end = datetime.now(timezone.utc)
    chunks = list(daterange_chunks(BACKFILL_START, end, CHUNK_DAYS))

    total_written = 0
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = None

        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            start_ts = int(chunk_start.timestamp())
            end_ts = int(chunk_end.timestamp())

            print(f"[{i}/{len(chunks)}] Fetching {chunk_start.date()} -> {chunk_end.date()}...", end=" ")

            try:
                raw_records = fetch_historical_raw(LAT, LON, api_key, start_ts, end_ts)
            except Exception as e:
                print(f"FAILED ({e}) — skipping this chunk")
                continue

            if not raw_records:
                print("no data returned")
                continue

            rows_written_this_chunk = 0
            for raw in raw_records:
                try:
                    features = compute_features(raw)
                except ValueError:
                    continue

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(features.keys()))
                    if not file_exists:
                        writer.writeheader()

                writer.writerow(features)
                rows_written_this_chunk += 1

            total_written += rows_written_this_chunk
            print(f"{rows_written_this_chunk} rows written")

            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. {total_written} total rows written to {OUTPUT_FILE}")


if __name__ == "__main__":
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set — check your .env file")

    run_backfill(api_key)