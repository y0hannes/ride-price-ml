import pandas as pd
import numpy as np

np.random.seed(42)  # Reproducible

n_rows = 200

data = {
    "distance_km": np.random.uniform(1, 30, n_rows),
    "duration_min": np.random.uniform(5, 60, n_rows),
    "time_of_day": np.random.choice(
        ["morning", "afternoon", "evening", "night"], n_rows
    ),
    "traffic_level": np.random.choice(["low", "medium", "high", "very_high"], n_rows),
    "weather": np.random.choice(["clear", "rainy", "cloudy", "stormy"], n_rows),
    "demand_level": np.random.choice(["low", "medium", "high", "peak"], n_rows),
    "surge_multiplier": np.random.uniform(1.0, 2.5, n_rows),
}

df = pd.DataFrame(data)

# Realistic price simulation
base_price = 2.0 * df["distance_km"] + 0.5 * df["duration_min"]
time_factor = np.where(np.isin(df["time_of_day"], ["evening", "night"]), 1.2, 1.0)
traffic_map = {"low": 1.0, "medium": 1.1, "high": 1.3, "very_high": 1.6}
weather_map = {"clear": 1.0, "rainy": 1.2, "cloudy": 1.1, "stormy": 1.5}
demand_map = {"low": 1.0, "medium": 1.1, "high": 1.3, "peak": 1.8}

df["traffic_factor"] = df["traffic_level"].map(traffic_map)
df["weather_factor"] = df["weather"].map(weather_map)
df["demand_factor"] = df["demand_level"].map(demand_map)

df["ride_price"] = base_price * time_factor * df["traffic_factor"] * df[
    "weather_factor"
] * df["demand_factor"] * df["surge_multiplier"] + np.random.normal(0, 2, n_rows)
df["ride_price"] = np.maximum(df["ride_price"], 25.0)  # Min 25 ETB

# Save cleaned dataset (only required features)
final_df = df[
    [
        "distance_km",
        "duration_min",
        "time_of_day",
        "traffic_level",
        "weather",
        "demand_level",
        "surge_multiplier",
        "ride_price",
    ]
]
final_df.to_csv("data/rides.csv", index=False)

print("- Dataset created: data/rides.csv")
print(f"- Shape: {final_df.shape}")
print(
    f"- Price range: {final_df['ride_price'].min():.1f} - {final_df['ride_price'].max():.1f} ETB"
)
print("\nSample:")
print(final_df.head())
