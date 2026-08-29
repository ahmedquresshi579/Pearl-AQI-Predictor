import pandas as pd
df = pd.read_csv("historical_features.csv")
before = len(df)
df = df.drop_duplicates(subset="timestamp", keep="first")
df.to_csv("historical_features.csv", index=False)
print(f"Removed {before - len(df)} duplicate rows. {len(df)} rows remain.")