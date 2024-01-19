import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import tkinter as tk
from tkinter import ttk



# Load the SARIMAX model
model_filename = 'sarimax_model.joblib'
loaded_model = joblib.load(model_filename)
sarimax_X_test = pd.read_csv('sarimax_X_test.csv')

# Load the model configuration
config_filename = 'sarimax_config.joblib'
loaded_config = joblib.load(config_filename)

# Load the min-max scaler used during training
# scaler_filename = 'scaler.joblib'
# scaler = joblib.load(scaler_filename)

end_date_test_data = sarimax_X_test.index[-1]

# Set up the Streamlit app
st.title("Oil Price Prediction")

input_features = {}

for param, value in loaded_config.items():
    # Provide default values for numerical input fields
    default_value = value if isinstance(value, (int, float)) else None
    user_input = st.sidebar.number_input(f"Enter {param}")
    input_features[param] = user_input

# Get the end date of the test data and set it as the min_value for the slider
forecast_steps = st.slider("Number of Forecast Steps", min_value=1, max_value=365, value=30, step=1)

# Generate future dates for the forecast
future_dates = pd.date_range(start=end_date_test_data, periods=forecast_steps + 1, freq='D')[1:]

# Create a DataFrame with the input features for each future date
future_data = pd.DataFrame(index=future_dates, data=[input_features] * forecast_steps)

# Concatenate the input_data and future_data for predictions
full_data = pd.concat([sarimax_X_test, future_data])

# Make predictions
forecast = loaded_model.get_forecast(steps=forecast_steps, exog=future_data)

# Display the predictions
# st.subheader("Predicted Prices:")
#st.write(forecast.predicted_mean)

# Plot the predictions over time
st.subheader("Predicted Prices Over Time:")
st.line_chart(forecast.predicted_mean)
