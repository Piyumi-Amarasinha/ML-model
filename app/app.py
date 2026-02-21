import streamlit as st
import pandas as pd
import joblib
import datetime
from pathlib import Path

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Agri-Price Predictor", layout="wide")
st.title("Live Carrot Price Predictor in Sri Lanka")
st.write("Enter the local climate and historical market conditions below to predict today's wholesale carrot price.")

# --- 2. LOAD THE MODEL & DATA STRUCTURE ---
try:
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "output"

    # Load the trained model
    model = joblib.load(output_dir / "xgboost_carrot_model.pkl")
    
    # Load the saved test data just to get the exact column names the model expects
    df_sample = pd.read_csv(output_dir / "prediction_results.csv")
    
    # Extract just the feature columns (ignore the target/prediction columns)
    expected_columns = [col for col in df_sample.columns if col not in ['Actual_Price', 'Predicted_Price']]
    
    # Extract the available regions from the column names (removing the "Region_" prefix)
    region_cols = [col for col in expected_columns if col.startswith('Region_')]
    # We add 'Ampara' back in manually because it was our drop_first=True baseline!
    available_regions = ['Ampara'] + [col.replace('Region_', '') for col in region_cols]

    def pick_expected_column(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in expected_columns:
                return candidate
        return None

    temp_col = pick_expected_column("Temperature_C", "Temperature (°C)", "Temperature (C)")
    rain_col = pick_expected_column("Rainfall_mm", "Rainfall (mm)")
    humidity_col = pick_expected_column("Humidity_pct", "Humidity (%)")
    impact_col = pick_expected_column("Crop_Yield_Impact_Score")
    price_7_col = pick_expected_column("Price_7_Days_Ago")
    year_col = pick_expected_column("Year")
    month_col = pick_expected_column("Month")
    dow_col = pick_expected_column("Day_of_Week")

    # --- 3. BUILD THE USER INPUT FORM ---
    st.subheader("Market Inputs")
    
    # Create two columns to make the form look professional
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

    # --- 4. THE PREDICTION BUTTON ---
    st.markdown("---")
    if st.button("🔮 Predict Carrot Price", type="primary"):
        
        # Create an empty dictionary with all expected features set to 0 initially
        input_data = dict.fromkeys(expected_columns, 0)
        
        # Fill in the continuous numerical variables
        missing_features: list[str] = []

        if temp_col is not None:
            input_data[temp_col] = temp
        else:
            missing_features.append("Temperature")

        if rain_col is not None:
            input_data[rain_col] = rainfall
        else:
            missing_features.append("Rainfall")

        if humidity_col is not None:
            input_data[humidity_col] = humidity
        else:
            missing_features.append("Humidity")

        if impact_col is not None:
            input_data[impact_col] = impact_score
        else:
            missing_features.append("Crop_Yield_Impact_Score")

        if price_7_col is not None:
            input_data[price_7_col] = price_7_days_ago
        else:
            missing_features.append("Price_7_Days_Ago")
        
        # Fill in the time variables extracted from the user's selected date
        if year_col is not None:
            input_data[year_col] = selected_date.year
        if month_col is not None:
            input_data[month_col] = selected_date.month
        if dow_col is not None:
            input_data[dow_col] = selected_date.weekday()

        if missing_features:
            st.warning(
                "Some expected feature columns were not found in the trained data, so the prediction may fail or be less accurate.\n\n"
                + "Missing: "
                + ", ".join(missing_features)
            )
        
        # Fill in the Region One-Hot Encoding
        # If they selected 'Ampara', all region columns remain 0 (which is exactly what the model expects for the baseline).
        if selected_region != 'Ampara':
            region_col_name = f"Region_{selected_region}"
            if region_col_name in input_data:
                input_data[region_col_name] = 1 # Set the specifically selected region to True (1)

        # Convert the dictionary into a Pandas DataFrame that the model can read
        input_df = pd.DataFrame([input_data])
        
        # Make the live prediction!
        predicted_price = model.predict(input_df)[0]
        
        # Display the result to the user
        st.success(f"### Predicted Wholesale Price: Rs. {predicted_price:.2f} per kg")

except FileNotFoundError:
    st.error(
        "Error: Could not find the model or data in the 'output' folder. "
        "Make sure you successfully ran TrainModel/trainmodel.py first!"
    )