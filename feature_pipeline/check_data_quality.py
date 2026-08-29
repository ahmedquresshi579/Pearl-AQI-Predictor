
import csv
from collections import Counter

with open("historical_features.csv") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")

aqi_vals = [float(r["aqi"]) for r in rows]
pm25_vals = [float(r["pm2_5"]) for r in rows]

print(f"AQI  — min: {min(aqi_vals):.1f}, max: {max(aqi_vals):.1f}, mean: {sum(aqi_vals)/len(aqi_vals):.1f}")
print(f"PM25 — min: {min(pm25_vals):.1f}, max: {max(pm25_vals):.1f}, mean: {sum(pm25_vals)/len(pm25_vals):.1f}")

neg_aqi = [v for v in aqi_vals if v < 0]
huge_pm25 = [v for v in pm25_vals if v > 1000]
print(f"Negative AQI values: {len(neg_aqi)}")
print(f"PM2.5 > 1000 (suspicious): {len(huge_pm25)}")

timestamps = [r["timestamp"] for r in rows]
dupes = len(timestamps) - len(set(timestamps))
print(f"Duplicate timestamps: {dupes}")

year_month = Counter(r["datetime_utc"][:7] for r in rows)
sorted_months = sorted(year_month.items())
print(f"Distinct months covered: {len(sorted_months)}")
print(f"Rows per month — min: {min(year_month.values())}, max: {max(year_month.values())}")

# flag any month with suspiciously low coverage (a month should have ~720-744 hourly rows)
sparse_months = [(m, c) for m, c in sorted_months if c < 400]
if sparse_months:
    print(f"\nMonths with <400 rows (likely gaps): {sparse_months}")
else:
    print("\nNo months with major gaps (all have 400+ hourly rows).")