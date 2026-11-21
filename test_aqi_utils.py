
from aqi_utils import classify_aqi, get_health_recommendation

test_values = [25, 75, 150, 250, 350, 450]

print("AQI CLASSIFICATION TEST")
print("-----------------------")

for value in test_values:
    category = classify_aqi(value)
    recommendation = get_health_recommendation(category)

    print(f"\nAQI: {value}")
    print(f"Category: {category}")
    print(f"Recommendation: {recommendation}")
