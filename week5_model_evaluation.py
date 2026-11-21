
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("dataset/processed/cleaned_city_day.csv")

# Select the same input features used during training
features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene"
]

X = df[features]
y = df["AQI"]

# Use the same train-test split as Week 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the saved Random Forest model
model = joblib.load("models/random_forest_aqi_model.pkl")

# Predict AQI for the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("WEEK 5 - RANDOM FOREST MODEL EVALUATION")
print("---------------------------------------")
print(f"Test records: {len(y_test)}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Display sample actual and predicted AQI values
results = pd.DataFrame({
    "Actual AQI": y_test.iloc[:10].values,
    "Predicted AQI": y_pred[:10].round(2)
})

print("\nSample Predictions:")
print(results.to_string(index=False))
