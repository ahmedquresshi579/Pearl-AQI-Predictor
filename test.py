import hopsworks

project = hopsworks.login(
    api_key_value="2Lh1WjXh4S4McYyO.sAdbVJJ6AabHxPFiBsDtnO1TUT4zSwFDcpI0Wzm9vdplXYcjige64kHVnrcc3E2O",
    project="lahore_aqi_ahmedanjum"
)
print("Connected to:", project.name)