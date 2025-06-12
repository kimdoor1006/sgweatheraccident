app_code = '''
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
st.markdown("Select a month between **2025 and 2026** to view forecasted weather conditions and predicted accident rate.")

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

    # --- Show weather summary in sentence format ---
    st.markdown("### 🌦️ Forecasted Weather Summary")

    st.markdown(f'''
    - **Total daily rainfall**: {input_row['total_rainfall']:.1f} mm  
    - **Mean temperature**: {input_row['mean_temp']:.1f} °C  
    - **Maximum temperature**: {input_row['max_temp']:.1f} °C  
    - **Minimum temperature**: {input_row['min_temp']:.1f} °C  
    - **Mean wind speed**: {input_row['mean_wind']:.1f} km/h  
    - **Maximum wind speed**: {input_row['max_wind']:.1f} km/h  
    ''')

except KeyError:
    st.error("❌ Data for the selected month is not available. Please choose another month.")
'''

# Save the updated script
with open("streamlit_app.py", "w") as f:
    f.write(app_code)

# Download it for deployment
from google.colab import files
files.download("streamlit_app.py")
