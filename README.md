# Machine Learning Training Model
This machine learning project predicts the daily wholesale price of carrots across different regions in Sri Lanka. By combining historical market prices with local climate data, the model provides short-term price forecasting to help farmers, traders, and stakeholders navigate agricultural market volatility.

✨ Key Features
Time-Series Forecasting: Utilizes a 7-day sliding window feature (Price_7_Days_Ago) to capture historical market trends and short-term price memory.

Climate Integration: Factors in daily temperature, rainfall, and humidity to adjust price predictions based on weather anomalies.

Explainable AI (XAI): Implements SHAP (SHapley Additive exPlanations) to break down the "black box" and visually explain how weather and history impact the final price.

Interactive Dashboard: Features a live Streamlit web application allowing users to input custom market conditions and receive instant predictions.

🗄️ Dataset & Preprocessing
The model is trained on a compiled dataset spanning 2020–2025, containing wholesale vegetable prices and climate records across 25 districts in Sri Lanka.

Preprocessing Steps Included:

Data Cleaning: Handled missing values and removed anomalous string characters from numeric columns (e.g., Crop Yield Impact Score).

Feature Engineering: Extracted temporal features (Year, Month, Day_of_Week) from raw Date strings to capture seasonal harvest cycles.

Lag Features: Shifted historical data to create the Price_7_Days_Ago predictor.

Encoding: Applied One-Hot Encoding to the categorical Region feature to ensure spatial data could be processed by the regression algorithm.

🧠 Model Architecture & Evaluation
To capture the complex, non-linear relationships between weather and agricultural economics without relying on deep learning, this project utilizes XGBoost (Extreme Gradient Boosting).

The model was evaluated using a strict chronological time-series split (80% Train / 20% Test) to prevent data leakage.

Evaluation Metrics Used: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared (R2).

📊 Explainability (SHAP)
Transparency is critical in financial and agricultural forecasting. SHAP values were extracted from the trained XGBoost model to ensure the algorithm's behavior aligns with real-world domain knowledge.

Feature Importance: Historical prices proved to be the strongest baseline predictor, followed closely by rainfall and temperature.

Summary Analysis: The SHAP beeswarm plots confirm that extreme weather events (like unusually high rainfall) correctly drive the model to predict higher wholesale prices due to implied harvest disruption.

🚀 How to Run Locally
1. Install Dependencies
Ensure you have Python installed, then install the required packages using pip:
pip install pandas numpy xgboost scikit-learn matplotlib shap streamlit joblib

2. Run the Dashboard
To launch the interactive frontend, run this command in your terminal:
streamlit run app.py

This will start a local server, and the dashboard will automatically open in your default web browser.
