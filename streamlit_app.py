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
st.title("🚧 Singapore Monthly Road Accident Forecast")
st.markdown("""
This application predicts the **total number of monthly road accidents in Singapore** using weather forecast data.  
Predictions are based on a machine learning model trained on **2011–2023 accident and weather records**.  
Select a future month between **2025 and 2026** to view the predicted accident count and associated weather summary.
""")

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
    st.metric("🚦 Predicted Monthly Accident Count", f"{predicted_accidents:.0f} cases")

    # --- Show weather summary in sentence format ---
    st.markdown("### 🌦️ Forecasted Weather Conditions")

    st.markdown(f'''
    - **Total monthly rainfall**: {input_row['total_rainfall']:.1f} mm  
    - **Mean temperature**: {input_row['mean_temp']:.1f} °C  
    - **Maximum temperature**: {input_row['max_temp']:.1f} °C  
    - **Minimum temperature**: {input_row['min_temp']:.1f} °C  
    - **Mean wind speed**: {input_row['mean_wind']:.1f} km/h  
    - **Maximum wind speed**: {input_row['max_wind']:.1f} km/h  
    ''')

# --- Model description ---
    st.markdown("### 🔍 Model Overview")
    st.info("""
    Predictions are generated using a **Gradient Boosting Regressor**, trained on weather-lag features.  
    The model was developed using data from 2011 to 2023 and evaluated for 2023 accident prediction performance.
    """)

    # --- Interpretation guidance ---
    st.markdown("### 📘 How to Interpret")
    st.markdown("""
    This forecast provides a **national-level estimate** of accident risk based on projected weather.  
    Elevated accident counts may coincide with months of **high rainfall**, **strong wind**, or **extreme temperature**.  
    Use this information for early planning, road safety campaigns, or risk mitigation during high-risk periods.
    """)

    # --- Disclaimer ---
    st.caption("""
    ⚠️ Note: This forecast does **not** account for traffic volume, road conditions, or driver behavior.  
    It represents a **statistical forecast based solely on weather patterns** and should be interpreted accordingly.
    """)

except KeyError:
    st.error("❌ Forecast data for the selected month is not available. Please choose another date.")
