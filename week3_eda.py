import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load cleaned dataset
df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

print("Shape:", df.shape)
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(8,5))
plt.hist(df["AQI"], bins=30)
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))

correlation = df.select_dtypes(include=['float64','int64']).corr()

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# AQI Category Count Plot

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="AQI_Bucket",
              order=df["AQI_Bucket"].value_counts().index)

plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Top 10 Most Polluted Cities
city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
city_aqi.plot(kind="bar")
plt.title("Top 10 Most Polluted Cities")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# Select features (X) and target (y)
X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
        'CO', 'SO2', 'O3', 'Benzene', 'Toluene']]

y = df['AQI']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)