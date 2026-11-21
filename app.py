import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from predict_aqi import predict_aqi
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
page_title="Smart Air Quality Monitor",
page_icon="🌿",
layout="wide"
)
# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "processed" / "cleaned_city_day.csv"
POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene"
]
DEFAULT_VALUES = {
    "PM2.5": 60.0,
    "PM10": 100.0,
    "NO": 15.0,
    "NO2": 25.0,
    "NOx": 35.0,
    "NH3": 10.0,
    "CO": 1.0,
    "SO2": 12.0,
    "O3": 30.0,
    "Benzene": 2.0,
    "Toluene": 5.0
}
AQI_CATEGORIES = [
    ("Good", "0–50", "#238b57"),
    ("Satisfactory", "51–100", "#a2b82d"),
    ("Moderate", "101–200", "#eba600"),
    ("Poor", "201–300", "#ef7620"),
    ("Very Poor", "301–400", "#cf2929"),
    ("Severe", "401+", "#7953bd")
]
# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
@st.cache_data
def load_dataset():
    """Load the cleaned historical air-quality dataset."""
    if not DATASET_PATH.exists():
        return None
    data = pd.read_csv(DATASET_PATH)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )
    return data
def get_aqi_color(category):
    """Return the display color for an AQI category."""
    colors = {
        "Good": "#238b57",
        "Satisfactory": "#a2b82d",
        "Moderate": "#eba600",
        "Poor": "#ef7620",
        "Very Poor": "#cf2929",
        "Severe": "#7953bd"
    }
    return colors.get(category, "#64748b")
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding:12px 0 8px 0;">
        <h1>🌿 Smart Air Quality Monitor</h1>
        <p style="color:gray;">
            AI-powered Air Quality Index prediction and health guidance
        </p>
    </div>
    """,
unsafe_allow_html=True
)
st.info(
    "Enter pollutant concentrations to estimate AQI. "
    "Use values and measurement conventions consistent with "
    "the original training dataset."
)
# --------------------------------------------------
# AQI CATEGORY GUIDE
# --------------------------------------------------
st.subheader("🎨 AQI Category Guide")
st.caption("Reference ranges used by this application's classifier.")
category_columns = st.columns(6)
for column, (name, limits, color) in zip(
    category_columns, AQI_CATEGORIES
):
    with column:
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                color:white;
                padding:12px 4px;
                border-radius:8px;
                text-align:center;
                min-height:65px;
            ">
                <b>{name}</b><br>
                <small>AQI {limits}</small>
            </div>
            """,
unsafe_allow_html=True
        )
# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 AQI Prediction",
    "📊 AQI Distribution",
    "🧪 Pollutant Comparison",
    "📈 Historical AQI Trend"
])
# ==================================================
# TAB 1: AQI PREDICTION
# ==================================================
with tab1:
    st.subheader("🧪 Pollution Parameters")
    st.caption(
        "Enter non-negative values for all 11 pollutants. "
        "The initial values are examples, not live measurements."
    )
    with st.form("aqi_prediction_form", enter_to_submit=False):
        st.markdown("### Particulate Matter")
        st.caption("Fine and coarse particles suspended in the air.")
        pm_col1, pm_col2 = st.columns(2)
        with pm_col1:
            pm25 = st.number_input(
                "PM2.5",
min_value=0.0,
value=DEFAULT_VALUES["PM2.5"],
step=1.0,
help="Enter the PM2.5 concentration."
            )
        with pm_col2:
            pm10 = st.number_input(
                "PM10",
min_value=0.0,
value=DEFAULT_VALUES["PM10"],
step=1.0,
help="Enter the PM10 concentration."
            )
        st.divider()
        st.markdown("### Gaseous Pollutants")
        col1, col2, col3 = st.columns(3)
        with col1:
            no = st.number_input(
                "NO", min_value=0.0,
value=DEFAULT_VALUES["NO"], step=1.0
            )
            no2 = st.number_input(
                "NO2", min_value=0.0,
value=DEFAULT_VALUES["NO2"], step=1.0
            )
            nox = st.number_input(
                "NOx", min_value=0.0,
value=DEFAULT_VALUES["NOx"], step=1.0
            )
        with col2:
            nh3 = st.number_input(
                "NH3", min_value=0.0,
value=DEFAULT_VALUES["NH3"], step=1.0
            )
            co = st.number_input(
                "CO", min_value=0.0,
value=DEFAULT_VALUES["CO"], step=0.1
            )
            so2 = st.number_input(
                "SO2", min_value=0.0,
value=DEFAULT_VALUES["SO2"], step=1.0
            )
        with col3:
            o3 = st.number_input(
                "O3", min_value=0.0,
value=DEFAULT_VALUES["O3"], step=1.0
            )
            benzene = st.number_input(
                "Benzene", min_value=0.0,
value=DEFAULT_VALUES["Benzene"], step=0.1
            )
            toluene = st.number_input(
                "Toluene", min_value=0.0,
value=DEFAULT_VALUES["Toluene"], step=0.1
            )
        st.divider()
        submitted = st.form_submit_button(
    "🔍 Predict Air Quality Index",
    width="stretch"
)
        
    if submitted:
        pollutants = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO": no,
            "NO2": no2,
            "NOx": nox,
            "NH3": nh3,
            "CO": co,
            "SO2": so2,
            "O3": o3,
            "Benzene": benzene,
            "Toluene": toluene
        }
        # Validate inputs before calling the model.
        validation_errors = []

        for pollutant, value in pollutants.items():
            if value < 0:
                validation_errors.append(
                    f"{pollutant} cannot be negative (entered: {value})."
                )

        # Use observed min/max from the cleaned training dataset as guardrails.
        # These limits come from project data rather than arbitrary constants.
        validation_data = load_dataset()
        if validation_data is not None:
            for pollutant, value in pollutants.items():
                if pollutant in validation_data.columns:
                    observed = pd.to_numeric(
                        validation_data[pollutant], errors="coerce"
                    ).dropna()
                    if not observed.empty:
                        minimum = float(observed.min())
                        maximum = float(observed.max())
                        if value < minimum or value > maximum:
                            validation_errors.append(
                                f"{pollutant} = {value:g} is outside the "
                                f"training dataset range ({minimum:g} to "
                                f"{maximum:g})."
                            )

        if validation_errors:
            st.error(
                "Prediction not generated. Please correct these input values:"
            )
            for message in validation_errors:
                st.write(f"- {message}")
            st.info(
                "The previous prediction, if shown, is from an earlier valid "
                "submission. Correct the values and click Predict again."
            )
        else:
            try:
                result = predict_aqi(pollutants)
                predicted_aqi = float(result["predicted_aqi"])
                category = result["category"]
                recommendation = result["recommendation"]
                color = get_aqi_color(category)
                st.divider()
                st.subheader("📊 AQI Prediction Result")
                st.markdown(
                    f"""
                    <div style="
                        background-color:{color};
                        color:white;
                        padding:25px 15px;
                        border-radius:12px;
                        text-align:center;
                        margin:10px 0 20px 0;
                    ">
                        <p style="margin:0;">Predicted Air Quality Index</p>
                        <h1 style="margin:8px 0;">
                            {predicted_aqi:.2f}
                        </h1>
                        <h3 style="margin:0;">{category}</h3>
                    </div>
                    """,
    unsafe_allow_html=True
                )
                # Website-only AQI alert based on the model's predicted value.
                st.subheader("🚨 AQI Alert")
                if predicted_aqi <= 50:
                    st.success(
                        "GOOD AIR QUALITY: Air quality is satisfactory. "
                        "Enjoy your usual outdoor activities."
                    )
                elif predicted_aqi <= 100:
                    st.info(
                        "SATISFACTORY AIR QUALITY: Air quality is acceptable. "
                        "Sensitive individuals should monitor how they feel."
                    )
                elif predicted_aqi <= 200:
                    st.warning(
                        "MODERATE AQI ALERT: Some pollutants may affect sensitive "
                        "individuals. Consider reducing prolonged outdoor exertion "
                        "if you experience discomfort."
                    )
                elif predicted_aqi <= 300:
                    st.warning(
                        "POOR AQI ALERT: Reduce prolonged or heavy outdoor exertion, "
                        "especially if you are sensitive to air pollution."
                    )
                elif predicted_aqi <= 400:
                    st.error(
                        "VERY POOR AQI ALERT: Avoid prolonged outdoor exertion. "
                        "Sensitive individuals should take extra precautions."
                    )
                else:
                    st.error(
                        "SEVERE AQI ALERT: Minimize outdoor exposure and follow "
                        "official local air-quality and health guidance."
                    )
                st.subheader("🩺 Health Recommendation")
                st.info(recommendation)
                st.caption(
                    "This is a model-generated estimate from the entered "
                    "values, not a live AQI measurement or medical diagnosis."
                )
                with st.expander("View Pollutant Input Summary"):
                    summary = pd.DataFrame({
                        "Pollutant": list(pollutants.keys()),
                        "Entered Value": list(pollutants.values())
                    })
                    st.dataframe(
                        summary,
    hide_index=True,
    width="stretch"
                    )
            except (ValueError, TypeError, KeyError) as error:
                st.error(f"Unable to generate prediction: {error}")
            except Exception as error:
                st.error(
                    "An unexpected error occurred while predicting AQI. "
                    "Check the terminal for details."
                )
                print(f"Prediction error: {error}")
    # ==================================================
# TAB 2: AQI DISTRIBUTION
# ==================================================
with tab2:
    st.subheader("Distribution of Air Quality Index")
    df = load_dataset()
    if df is None:
        st.warning(
            "Historical dataset not found. Place cleaned_city_day.csv "
            "inside dataset/processed/ to view this chart."
        )
    elif "AQI" not in df.columns:
        st.error("The dataset does not contain an AQI column.")
    else:
        aqi_values = pd.to_numeric(
            df["AQI"], errors="coerce"
        ).dropna()
        if aqi_values.empty:
            st.warning("No valid AQI values are available.")
        else:
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric(
                    "Average AQI",
                    f"{aqi_values.mean():.2f}"
                )
            with metric2:
                st.metric(
                    "Median AQI",
                    f"{aqi_values.median():.2f}"
                )
            with metric3:
                st.metric(
                    "Total Observations",
                    f"{len(aqi_values):,}"
                )
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(
                aqi_values,
bins=30,
edgecolor="black"
            )
            ax.set_title("Distribution of Air Quality Index (AQI)")
            ax.set_xlabel("AQI Value")
            ax.set_ylabel("Number of Observations")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "The histogram shows how frequently different AQI "
                "values occur in the historical dataset."
            )
# ==================================================
# TAB 3: POLLUTANT COMPARISON
# ==================================================
with tab3:
    st.subheader("Average Concentration of Air Pollutants")
    df = load_dataset()
    if df is None:
        st.warning(
            "Historical dataset not found. Place cleaned_city_day.csv "
            "inside dataset/processed/ to view this chart."
        )
    else:
        available_pollutants = [
            pollutant for pollutant in POLLUTANTS
            if pollutant in df.columns
        ]
        if not available_pollutants:
            st.error("No expected pollutant columns were found.")
        else:
            average_values = (
                df[available_pollutants]
                .apply(pd.to_numeric, errors="coerce")
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )
            if average_values.empty:
                st.warning("No valid pollutant measurements are available.")
            else:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                average_values.plot(
kind="bar",
ax=ax2,
edgecolor="black"
                )
                ax2.set_title(
                    "Average Concentration of Air Pollutants"
                )
                ax2.set_xlabel("Pollutant")
                ax2.set_ylabel(
                    "Average Concentration (dataset units)"
                )
                ax2.tick_params(axis="x", rotation=45)
                ax2.grid(axis="y", alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption(
                    "Pollutants can use different measurement units. "
                    "Bar heights should not be interpreted as a direct "
                    "comparison of health risk."
                )
                with st.expander("View Average Concentrations"):
                    average_table = average_values.rename(
                        "Average Concentration"
                    ).reset_index()
                    average_table.columns = [
                        "Pollutant",
                        "Average Concentration"
                    ]
                    st.dataframe(
                    average_table.round(3),
    hide_index=True,
    width="stretch"
)
# ==================================================
# TAB 4: HISTORICAL AQI TREND
# ==================================================
with tab4:
    st.subheader("Historical Monthly Average AQI Trend")
    df = load_dataset()
    if df is None:
        st.warning(
            "Historical dataset not found. Place cleaned_city_day.csv "
            "inside dataset/processed/ to view this chart."
        )
    elif not {"Date", "AQI"}.issubset(df.columns):
        st.error("The dataset must contain both Date and AQI columns.")
    else:
        trend_data = df[["Date", "AQI"]].copy()
        trend_data["Date"] = pd.to_datetime(
            trend_data["Date"], errors="coerce"
        )
        trend_data["AQI"] = pd.to_numeric(
            trend_data["AQI"], errors="coerce"
        )
        trend_data = trend_data.dropna(
subset=["Date", "AQI"]
        )
        if trend_data.empty:
            st.warning("No valid date and AQI records are available.")
        else:
            trend_data["Month"] = (
                trend_data["Date"].dt.to_period("M")
            )
            monthly_aqi = (
                trend_data.groupby("Month")["AQI"]
                .mean()
                .sort_index()
            )
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(
                monthly_aqi.index.astype(str),
                monthly_aqi.values,
marker="o",
linewidth=2
            )
            ax3.set_title("Historical Monthly Average AQI Trend")
            ax3.set_xlabel("Month")
            ax3.set_ylabel("Average AQI")
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3)
            # Show fewer date labels to keep the chart readable.
            tick_step = max(1, len(monthly_aqi) // 12)
            tick_positions = list(
                range(0, len(monthly_aqi), tick_step)
            )
            ax3.set_xticks(tick_positions)
            ax3.set_xticklabels(
                monthly_aqi.index.astype(str)[tick_positions],
rotation=45,
ha="right"
            )
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            st.caption(
                "This chart displays historical monthly average AQI "
                "from the dataset. It is not a future AQI forecast."
            )
            with st.expander("View Monthly AQI Data"):
                monthly_table = pd.DataFrame({
                    "Month": monthly_aqi.index.astype(str),
                    "Average AQI": monthly_aqi.values
                })
                st.dataframe(
    monthly_table.round(2),
    hide_index=True,
    width="stretch"
)
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "AI-Driven Smart Air Quality and Pollution Monitoring System "
    "| Machine Learning Project"
)
