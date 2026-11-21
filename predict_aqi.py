
import joblib
import pandas as pd

from aqi_utils import classify_aqi, get_health_recommendation

# Load the saved model
model = joblib.load("models/random_forest_aqi_model.pkl")

# Input features must match the model's training order
FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene"
]


def predict_aqi(pollutants):
    """
    Predict AQI from a dictionary containing all 11 pollutant values.
    """

    missing = [feature for feature in FEATURES if feature not in pollutants]
    if missing:
        raise ValueError(f"Missing pollutant values: {missing}")

    values = [float(pollutants[feature]) for feature in FEATURES]

    if any(value < 0 for value in values):
        raise ValueError("Pollutant values cannot be negative.")

    input_data = pd.DataFrame([values], columns=FEATURES)

    predicted_aqi = float(model.predict(input_data)[0])
    predicted_aqi = max(0.0, predicted_aqi)

    # Round to the nearest integer for category classification
    category = classify_aqi(round(predicted_aqi))
    recommendation = get_health_recommendation(category)

    return {
        "predicted_aqi": round(predicted_aqi, 2),
        "category": category,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    sample_pollutants = {
        "PM2.5": 150,
        "PM10": 250,
        "NO": 15,
        "NO2": 25,
        "NOx": 35,
        "NH3": 10,
        "CO": 1.0,
        "SO2": 12,
        "O3": 30,
        "Benzene": 2,
        "Toluene": 5
    }

    result = predict_aqi(sample_pollutants)

    print("AQI PREDICTION RESULT")
    print("---------------------")
    print(f"Predicted AQI: {result['predicted_aqi']}")
    print(f"AQI Category: {result['category']}")
    print(f"Health Recommendation: {result['recommendation']}")
