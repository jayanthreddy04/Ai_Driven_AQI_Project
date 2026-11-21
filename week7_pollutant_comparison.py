
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

# Pollutants used by the AQI prediction model
pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

# Check that all required columns exist
missing_columns = [
    column for column in pollutants
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing pollutant columns: {missing_columns}"
    )

# Calculate average concentration of each pollutant
average_values = (
    df[pollutants]
    .apply(pd.to_numeric, errors="coerce")
    .mean()
    .sort_values(ascending=False)
)

# Create the comparison chart
plt.figure(figsize=(11, 6))
average_values.plot(kind="bar", edgecolor="black")

plt.title("Average Concentration of Air Pollutants")
plt.xlabel("Pollutant")
plt.ylabel("Average Concentration (dataset units)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save the chart
plt.savefig(
    "pollutant_comparison_week7.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Pollutant comparison chart created successfully!")
print("\nAverage pollutant values:")
print(average_values.round(2))
