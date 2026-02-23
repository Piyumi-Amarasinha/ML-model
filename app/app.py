from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="Agri-Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("ℹ️ About this Application")
st.sidebar.info(
    "**Live Carrot Price Predictor**\n\n"
    "This machine learning dashboard predicts the daily wholesale price of carrots across 25 districts in Sri Lanka. "
    "It uses an **XGBoost Regressor** trained on 130,000 historical records from 2020 to 2025.\n\n"
    "**Key Predictive Features:**\n"
    "- 7-Day Historical Price Lag\n"
    "- Weather anomalies (Rainfall, Temp, Humidity)\n"
    "- Regional Market Differences\n\n"
    "*Developed as a Machine Learning academic project focusing on local dataset compilation, advanced algorithm selection, and Explainable AI.*"
)

st.title("🥕 Live Carrot Price Predictor")
st.write("Enter the local climate and historical market conditions below to predict today's wholesale carrot price.")


def _set_first_existing(input_row: dict, candidates: list[str], value) -> None:
    for candidate in candidates:
        if candidate in input_row:
            input_row[candidate] = value
            return
    raise KeyError(
        f"None of the expected feature columns exist: {candidates}. "
        "Re-train the model or update the app feature mapping."
    )


try:
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "output" / "xgboost_carrot_model.pkl"
    sample_path = base_dir / "output" / "prediction_results.csv"

    model = joblib.load(model_path)
    df_sample = pd.read_csv(sample_path)

    expected_columns = [col for col in df_sample.columns if col not in ["Actual_Price", "Predicted_Price"]]
    region_cols = [col for col in expected_columns if col.startswith("Region_")]
    available_regions = ["Ampara"] + [col.replace("Region_", "") for col in region_cols]

    st.subheader("Market Inputs")
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input("Select Target Date", datetime.date.today())
        selected_region = st.selectbox("Select Region", sorted(available_regions))
        price_7_days_ago = st.number_input("Carrot Price 7 Days Ago (LKR/kg)", value=250.0, step=10.0)

    with col2:
        temp = st.number_input("Average Temperature (°C)", value=30.0, step=0.5)
        rainfall = st.number_input("Rainfall (mm)", value=15.0, step=1.0)
        humidity = st.number_input("Humidity (%)", value=75.0, step=1.0)
        impact_score = st.number_input("Crop Yield Impact Score (0.0 - 2.0)", value=1.50, step=0.1)

    st.markdown("---")
    if st.button("🔮 Predict Carrot Price", type="primary"):
        input_data = {col: 0 for col in expected_columns}

        _set_first_existing(input_data, ["Temperature (°C)", "Temperature_C"], temp)
        _set_first_existing(input_data, ["Rainfall_mm"], rainfall)
        _set_first_existing(input_data, ["Humidity_pct"], humidity)
        _set_first_existing(input_data, ["Crop_Yield_Impact_Score"], impact_score)
        _set_first_existing(input_data, ["Price_7_Days_Ago"], price_7_days_ago)
        _set_first_existing(input_data, ["Year"], selected_date.year)
        _set_first_existing(input_data, ["Month"], selected_date.month)
        _set_first_existing(input_data, ["Day_of_Week"], selected_date.weekday())

        if selected_region != "Ampara":
            region_col_name = f"Region_{selected_region}"
            if region_col_name in input_data:
                input_data[region_col_name] = 1

        input_df = pd.DataFrame([input_data])[expected_columns]
        predicted_price = model.predict(input_df)[0]
        st.success(f"### 📈 Predicted Wholesale Price: Rs. {predicted_price:.2f} per kg")

        st.markdown("---")
        st.subheader("📋 Input Summary")
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        sum_col1.metric("Target Date", str(selected_date))
        sum_col1.metric("Region", selected_region)
        sum_col2.metric("Price 7 Days Ago", f"Rs. {price_7_days_ago:.2f}")
        sum_col2.metric("Temperature", f"{temp} °C")
        sum_col3.metric("Rainfall", f"{rainfall} mm")
        sum_col3.metric("Humidity", f"{humidity} %")

        st.markdown("---")
        st.subheader("📊 Prediction Explanation (SHAP Waterfall Plot)")
        st.write(
            "This waterfall plot breaks down how your specific inputs pushed the prediction "
            "up or down from the baseline."
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)

        fig = plt.figure(figsize=(4, 3))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("🌐 Global Model Analysis")
        st.write("These charts explain what the model learned from the entire dataset.")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.write("**Feature Importance (Bar Plot)**")
            image_path = base_dir / "shap_feature_importance.png"
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Could not find shap_feature_importance.png in the project folder.")

        with img_col2:
            st.write("**SHAP Beeswarm Summary**")
            image_path = base_dir / "shap_summary_plot.png"
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Could not find shap_summary_plot.png in the project folder.")

except FileNotFoundError as e:
    st.error(
        "Error: Could not find the model or data in the project 'output' folder. "
        "Make sure you successfully ran TrainModel/trainmodel.py first.\n\n"
        f"Details: {e}"
    )