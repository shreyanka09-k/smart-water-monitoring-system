
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Water Monitoring",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "temporal_random_forest.joblib")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, config


model, config = load_model()

THRESHOLD = config["threshold"]
FEATURES = config["features"]


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def create_temporal_features(data):

    data = data.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    temporal_features = [
        "gw_lvl_site2_anomaly_score",
        "gw_temp_site2_anomaly_score",
        "temp_site1_anomaly_score",
        "sfc_temp_site1_anomaly_score"
    ]

    # Time features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month_num"] = data["timestamp"].dt.month

    # Cyclical encoding
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # Lag, rolling and change features
    for col in temporal_features:

        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag6"] = data[col].shift(6)
        data[f"{col}_lag24"] = data[col].shift(24)

        past = data[col].shift(1)

        data[f"{col}_rolling_mean_6h"] = (
            past.rolling(6, min_periods=1).mean()
        )

        data[f"{col}_rolling_mean_24h"] = (
            past.rolling(24, min_periods=1).mean()
        )

        data[f"{col}_rolling_std_24h"] = (
            past.rolling(24, min_periods=2).std()
        )

        data[f"{col}_change_1h"] = (
            data[col] - data[col].shift(1)
        )

        data[f"{col}_change_24h"] = (
            data[col] - data[col].shift(24)
        )

    return data


# ============================================================
# HEADER
# ============================================================

st.title("💧 Smart Water Monitoring System")

st.markdown(
    """
    **AI-based monitoring of sensor patterns associated with
    historical leak-proximity periods.**
    """
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Water Sensor CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded: {len(data):,} observations"
        )

        # Create temporal features
        processed_data = create_temporal_features(data)

        # Check required features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in processed_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded dataset is missing required "
                f"features: {missing_features}"
            )

            st.stop()

        X_new = processed_data[FEATURES]

        # ====================================================
        # PREDICTION
        # ====================================================

        probabilities = model.predict_proba(X_new)[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        results = pd.DataFrame({
            "timestamp": processed_data["timestamp"],
            "risk_score": probabilities,
            "prediction": predictions
        })

        results["status"] = results["prediction"].map({
            0: "Normal",
            1: "Potential Leak-Proximity"
        })

        # ====================================================
        # SUMMARY
        # ====================================================

        total = len(results)
        potential = int(results["prediction"].sum())
        normal = total - potential
        average_risk = results["risk_score"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Observations",
            f"{total:,}"
        )

        col2.metric(
            "Potential Leak-Proximity",
            f"{potential:,}"
        )

        col3.metric(
            "Normal",
            f"{normal:,}"
        )

        col4.metric(
            "Average Risk Score",
            f"{average_risk:.1%}"
        )

        st.divider()

        # ====================================================
        # LATEST STATUS
        # ====================================================

        latest = results.iloc[-1]

        st.subheader("Latest Monitoring Status")

        if latest["prediction"] == 1:

            st.warning(
                f"⚠️ Potential Leak-Proximity detected\n\n"
                f"Risk score: {latest['risk_score']:.1%}"
            )

        else:

            st.success(
                f"✅ Normal\n\n"
                f"Risk score: {latest['risk_score']:.1%}"
            )

        # ====================================================
        # RISK TREND
        # ====================================================

        st.subheader("Risk Trend")

        chart_data = results[
            ["timestamp", "risk_score"]
        ].set_index("timestamp")

        st.line_chart(chart_data)

        # ====================================================
        # HIGH-RISK OBSERVATIONS
        # ====================================================

        st.subheader("Highest-Risk Observations")

        high_risk = (
            results
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            high_risk,
            use_container_width=True
        )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        csv_output = results.to_csv(index=False)

        st.download_button(
            "⬇️ Download Predictions",
            data=csv_output,
            file_name="water_monitoring_predictions.csv",
            mime="text/csv"
        )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander("Model Information"):

            st.write(
                f"**Model:** {config['model_name']}"
            )

            st.write(
                f"**Number of features:** {len(FEATURES)}"
            )

            st.write(
                f"**Decision threshold:** {THRESHOLD}"
            )

            st.write(
                "**Temporal features:** "
                "lags, rolling statistics, changes and "
                "cyclical time features."
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "This system predicts patterns associated with "
            "historical leak-proximity labels. A "
            "'Potential Leak-Proximity' prediction does not "
            "confirm a physical leak and should be followed "
            "by operational inspection."
        )

    except Exception as e:

        st.error(
            f"Error processing the uploaded file: {e}"
        )

else:

    st.info(
        "Upload a water sensor CSV file to begin monitoring."
    )
