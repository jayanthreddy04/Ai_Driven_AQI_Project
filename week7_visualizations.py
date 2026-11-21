
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

# Check that the AQI column exists
if "AQI" not in df.columns:
    raise ValueError("The dataset does not contain an AQI column.")

# Convert AQI to numeric and remove invalid values
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
aqi_values = df["AQI"].dropna()

# Plot AQI distribution
plt.figure(figsize=(10, 6))
plt.hist(aqi_values, bins=40, edgecolor="black")

plt.title("Distribution of Air Quality Index (AQI)")
plt.xlabel("AQI Value")
plt.ylabel("Number of Observations")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save the chart
plt.savefig("aqi_distribution_week7.png", dpi=150)
plt.show()

print("AQI distribution chart created successfully!")
print(f"Total valid AQI records: {len(aqi_values)}")
