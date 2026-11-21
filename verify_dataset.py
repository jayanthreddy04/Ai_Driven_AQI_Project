import pandas as pd

df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())