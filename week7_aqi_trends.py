
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

# Check required columns
required_columns = ["Date", "AQI"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Convert columns to appropriate data types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

# Remove invalid records
df = df.dropna(subset=["Date", "AQI"])

# Calculate average AQI for each month
df["YearMonth"] = df["Date"].dt.to_period("M")
monthly_aqi = (
    df.groupby("YearMonth")["AQI"]
    .mean()
    .sort_index()
)

# Convert monthly periods to timestamps for plotting
monthly_dates = monthly_aqi.index.to_timestamp()

# Plot historical AQI trend
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_dates,
    monthly_aqi.values,
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.title("Historical Monthly Average AQI Trend")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save chart
plt.savefig(
    "historical_aqi_trends_week7.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Historical AQI trend chart created successfully!")
print("\nFirst few monthly averages:")
print(monthly_aqi.head().round(2))
print(f"\nMonths analyzed: {len(monthly_aqi)}")
