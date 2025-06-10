
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Load model and scaler ---
model = joblib.load("gb_accident_model.pkl")
scaler = joblib.load("accident_scaler.pkl")

# --- Load forecast data ---
forecast = pd.read_csv("final_forecast.csv", parse_dates=["year_month"])
forecast.set_index("year_month", inplace=True)

# --- App title and instructions ---
st.title("🚧 Road Accident Rate Forecast")
st.markdown("Select a month between **2025 and 2027** to view forecasted weather conditions and predicted accident rate.")

# --- Date selector ---
selected_date = st.date_input("📅 Select a future month:", min_value=datetime(2025, 1, 1), max_value=datetime(2027, 1, 1))

# --- Prepare input for prediction ---
try:
    selected_ym = pd.to_datetime(f"{selected_date.year}-{selected_date.month}-01")
    input_row = forecast.loc[selected_ym]

    # Use only lag-based features that the model was trained on
    lag_features = [col for col in forecast.columns if '_lag_' in col]
    X_input = input_row[lag_features].values.reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Predict
    predicted_accidents = model.predict(X_scaled)[0]

    # --- Display result ---
    st.subheader(f"📊 Accident Forecast for {selected_ym.strftime('%B %Y')}")
    st.metric("🚦 Estimated Monthly Accidents", f"{predicted_accidents:.0f} cases")

    # --- Show weather summary (non-lag values) ---
    st.markdown("### 🌦️ Forecasted Weather")
    weather_cols = [col for col in forecast.columns if '_lag_' not in col and col != "accident_count"]
    st.dataframe(input_row[weather_cols].to_frame().T)

except KeyError:
    st.error("❌ Data for the selected month is not available. Please choose another month.")
