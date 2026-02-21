from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Loading dataset...")
base_dir = Path(__file__).resolve().parents[1]
dataset_path = base_dir / "Preprocess" / "cleaned_carrot_prices_for_ML.csv"
df = pd.read_csv(dataset_path)

# 1. Ensure the data is strictly chronological for time-series predicting
df = df.sort_values(by=['Year', 'Month', 'Date'])

# XGBoost requires all data to be numbers. 
# We convert our One-Hot Encoded True/False columns into 1s and 0s.
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# Drop the 'Date' string column since the model only reads numbers 
# (We already extracted Month, Day_of_Week, and Year earlier)
df = df.drop(columns=['Date'])

# 2. Define Features (X) and Target (y)
X = df.drop(columns=['Vegetable_Price_LKR_kg'])
y = df['Vegetable_Price_LKR_kg']

# 3. Time-Series Train/Test Split (80% Train, 20% Test)
# We CANNOT use a random split here. If we did, the model would peek into the future to predict the past.
# We must train on the oldest 80% of data, and test on the newest 20%.
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training on {len(X_train)} historical records...")
print(f"Testing on {len(X_test)} future records...")

# 4. Initialize and Train the XGBoost Model (Hyperparameters)
# n_estimators: Number of sequential trees built.
# learning_rate: Controls how aggressively each tree corrects the last one to prevent overfitting.
# max_depth: How deep each decision tree is allowed to go.
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
r = np.corrcoef(y_test, predictions)[0, 1]

print("\n--- MODEL PERFORMANCE METRICS ---")
# MAE: Tells you the average LKR difference between your prediction and the actual price
print(f"Mean Absolute Error (MAE): Rs. {mae:.2f}")
# RMSE: Penalizes the model heavily for making massive incorrect predictions
print(f"Root Mean Squared Error (RMSE): Rs. {rmse:.2f}")
print(f"R (Correlation): {r:.4f}")
print(f"R^2 (R-squared): {r2:.4f}")

# 6. Plot the Results (Required for your report)
plt.figure(figsize=(10, 5))
# Plotting just the first 100 predictions so the graph is readable
plt.plot(y_test.values[:100], label='Actual Price', color='blue', marker='o')
plt.plot(predictions[:100], label='Predicted Price', color='red', linestyle='dashed', marker='x')
plt.title('XGBoost Predictions vs Actual Carrot Prices (First 100 Test Samples)')
plt.xlabel('Time (Test Samples)')
plt.ylabel('Price (LKR/kg)')
plt.legend()
plt.tight_layout()

# Save the plot instead of showing it
plot_path = base_dir / "xgboost_predictions_plot.png"
plt.savefig(plot_path)
print(f"\nSuccess! A visual graph of the predictions has been saved as '{plot_path}'")

import shap
import matplotlib.pyplot as plt

print("\nCalculating SHAP values (this might take a few seconds)...")

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar)')
plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.clf()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot (Impact on Carrot Price)')
plt.tight_layout()
plt.savefig('shap_summary_plot.png')

print("Success! SHAP plots have been saved as 'shap_feature_importance.png' and 'shap_summary_plot.png'.")

import os
import joblib

print("\nSaving model and data for the frontend...")

# 1. Create an 'output' folder if it doesn't already exist (always at project root)
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Save the trained XGBoost model
joblib.dump(model, output_dir / "xgboost_carrot_model.pkl")

# 3. Save the test data alongside the actual and predicted prices
results_df = X_test.copy()
results_df['Actual_Price'] = y_test
results_df['Predicted_Price'] = predictions

results_df.to_csv(output_dir / "prediction_results.csv", index=False)

print("Success! Model and data saved to the 'output' folder.")