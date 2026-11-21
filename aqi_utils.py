
def classify_aqi(aqi):
    """Return the Indian AQI category for a predicted AQI value."""

    if aqi < 0:
        return "Invalid AQI"

    elif aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


def get_health_recommendation(category):
    """Return a general health recommendation for the AQI category."""

    recommendations = {
        "Good": "Air quality is good. Enjoy your usual outdoor activities.",
        "Satisfactory": "Sensitive individuals should consider reducing prolonged outdoor exertion.",
        "Moderate": "People with asthma or respiratory conditions should limit prolonged outdoor exertion.",
        "Poor": "Reduce prolonged outdoor activities, especially if you experience breathing discomfort.",
        "Very Poor": "Avoid prolonged outdoor exertion and reduce exposure to polluted air.",
        "Severe": "Avoid outdoor exertion as much as possible and follow local health advisories.",
        "Invalid AQI": "Please provide a valid, non-negative AQI value."
    }

    return recommendations.get(
        category,
        "AQI category unavailable. Please check the input."
    )
