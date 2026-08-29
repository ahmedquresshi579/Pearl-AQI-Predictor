
# EPA breakpoints: (Conc_low, Conc_high, AQI_low, AQI_high)
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 400),
    (325.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """
    Convert a PM2.5 concentration (µg/m³) to US EPA AQI (0-500).
    Values above the top breakpoint are clamped to 500 (hazardous, off-scale).
    Negative/invalid values raise, since they indicate a data problem
    upstream that should NOT be silently converted into a fake AQI.
    """
    if pm25 is None:
        raise ValueError("pm25_to_aqi received None — check upstream data before calling this")
    if pm25 < 0:
        raise ValueError(f"pm25_to_aqi received a negative value ({pm25}) — bad data upstream")

    for bp_low, bp_high, aqi_low, aqi_high in PM25_BREAKPOINTS:
        if bp_low <= pm25 <= bp_high:
            return round(
                ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low,
                1,
            )

    # Above the highest defined breakpoint (500.4) — clamp, don't extrapolate.
    # Extrapolating past the EPA's own table produces numbers with no real meaning.
    if pm25 > PM25_BREAKPOINTS[-1][1]:
        return 500.0

    raise ValueError(f"pm25_to_aqi: {pm25} did not match any breakpoint — check input")


if __name__ == "__main__":
    # Quick sanity checks against known values
    test_cases = [
        (0, 0),
        (9.0, 50),
        (38.89, 109.3),   # today's real Lahore reading, validated earlier
        (500.4, 500),
        (600, 500),        # clamp test
    ]
    for pm25, expected in test_cases:
        result = pm25_to_aqi(pm25)
        status = "OK" if abs(result - expected) < 1.0 else "MISMATCH"
        print(f"PM2.5={pm25:>7} -> AQI={result:>6} (expected ~{expected}) [{status}]")