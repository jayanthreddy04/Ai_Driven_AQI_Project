import pandas as pd

# Load dataset
df = pd.read_csv("dataset/raw/city_day.csv")

# Basic information
print("Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])

print("\nShape after removing missing AQI rows:")
print(df.shape)

print("\nMissing values after removing missing AQI:")
print(df.isnull().sum())


# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Drop Xylene column
df = df.drop(columns=["Xylene"])

# Fill remaining missing values with median
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("dataset/processed/cleaned_city_day.csv", index=False)

print("\nCleaned dataset saved successfully!")