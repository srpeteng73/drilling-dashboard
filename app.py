import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import matplotlib.pyplot as plt
import os # Added for environment variables
import smtplib # Added for email functionality
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ML/AI Libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Page Configuration ---
st.set_page_config(
    page_title="Drilling & Maintenance Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (No Changes) ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keeping CSS brief for clarity

# --- SESSION STATE INITIALIZATION ---
# For Live Drilling Dashboard
if 'drilling_data' not in st.session_state: st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
if 'session_history' not in st.session_state: st.session_state.session_history = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
# For Predictive Maintenance Email Alert
if 'summary_sent' not in st.session_state: st.session_state['summary_sent'] = False

# --- PREDICTIVE MAINTENANCE HELPER FUNCTIONS ---
@st.cache_data
def generate_full_field_data(asset_ids, start_time='2025-08-01', total_periods=200):
    all_asset_data = []
    for asset_id in asset_ids:
        time_index = pd.date_range(start=start_time, periods=total_periods, freq='H')
        temp = np.random.normal(loc=75, scale=2, size=total_periods)
        vibe = np.random.normal(loc=0.3, scale=0.05, size=total_periods)
        anomaly_start = np.random.randint(total_periods // 2, total_periods - 20)
        temp[anomaly_start:anomaly_start+10] += np.linspace(5, 25, 10)
        vibe[anomaly_start:anomaly_start+10] += np.linspace(0.1, 0.6, 10)
        all_asset_data.append(pd.DataFrame({'timestamp': time_index, 'asset_id': asset_id, 'temperature': temp, 'vibration': vibe}))
    return pd.concat(all_asset_data, ignore_index=True)

@st.cache_resource
def train_anomaly_model(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    features = data[['temperature', 'vibration']]
    model.fit(features)
    return model

def predict_anomalies(model, data):
    features = data[['temperature', 'vibration']]
    data['is_anomaly'] = model.predict(features)
    data['is_anomaly'] = data['is_anomaly'].map({1: 0, -1: 1})
    return data

@st.cache_resource
def train_lstm_model(data, feature='temperature'):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(data[[feature]])
    X_train, y_train = [], []
    for i in range(len(scaled_series) - 10):
        X_train.append(scaled_series[i:i+10])
        y_train.append(scaled_series[i+10])
    X_train, y_train = np.array(X_train), np.array(y_train)
    lstm_model = Sequential([LSTM(50, activation='relu', input_shape=(10, 1)), Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=1, verbose=0)
    return lstm_model, scaler

def predict_next_hour_temp(model, scaler, data, feature='temperature'):
    scaled_series = scaler.transform(data[[feature]])
    last_sequence = scaled_series[-10:].reshape(1, 10, 1)
    predicted_value = model.predict(last_sequence)
    return scaler.inverse_transform(predicted_value)[0][0]

def send_summary_email(alerts_df):
    try:
        sender_email = os.getenv("EMAIL_USERNAME")
        password = os.getenv("EMAIL_PASSWORD")
        if not sender_email or not password:
            st.error("Email credentials are not configured. Please set environment variables.")
            return
        recipient_email = sender_email
        subject = f"🚨 ({len(alerts_df)}) Critical Alerts Detected in the Oilfield"
        body = f"Hello Operations Team,\n\nThe following critical alerts require attention:\n\n{alerts_df.to_string(index=False)}\n\nPlease inspect the assets."
        msg = MIMEMultipart(); msg['From'] = sender_email; msg['To'] = recipient_email; msg['Subject'] = subject; msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(sender_email, password); server.send_message(msg); server.quit()
        st.toast(f"✅ Summary email with {len(alerts_df)} alerts sent successfully!")
    except Exception as e:
        st.error(f"Failed to send summary email. Error: {e}")

# --- SIDEBAR LAYOUT ---
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")
st.sidebar.header("Drilling Suite Controls")
rpm_mean = st.sidebar.slider("Live Target RPM", 100, 200, 150)
vibration_threshold = st.sidebar.slider("Live Vibration Threshold", 0.8, 1.5, 1.0)
sim_rpm = st.sidebar.slider("Simulated RPM", 30, 300, 120)
sim_wob = st.sidebar.slider("Simulated WOB (kN)", 10, 100, 50)

st.sidebar.markdown("---")
st.sidebar.header("Predictive Maintenance Settings")
temp_threshold = st.sidebar.slider("Temperature Threshold (°F)", 80, 120, 95)
vib_threshold = st.sidebar.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)
downtime_cost = st.sidebar.slider("Downtime Cost per Hour ($)", 500, 5000, 1000, step=100)

# --- MAIN APPLICATION TITLE ---
st.title("Comprehensive Drilling & Asset Management Suite")

# --- CREATE TABS ---
drilling_tabs = ["**Executive Summary**", "**📖 User Guide**", "**📊 Live Drilling**", "**⚙️ Drilling Simulator**"]
maintenance_tabs = ["**🔧 Asset Health Overview**", "**🤖 AI Forecast & Anomalies**", "**📅 Maintenance Plan**"]
all_tabs = drilling_tabs + maintenance_tabs
selected_tabs = st.tabs(all_tabs)

# --- Drilling Tabs ---
with selected_tabs[0]: # Summary
    st.header("An Integrated Solution for Modern Oilfield Operations")
    st.markdown('<p class="guide-text">Developed by <strong>Mr. Omar Nur, a Petroleum Engineer</strong>, this application suite provides a holistic view of oilfield management, from real-time drilling optimization to full-field predictive asset maintenance.</p>', unsafe_allow_html=True)
# ... other drilling tabs ... (code truncated for brevity, but it's all there from previous steps)

# --- Predictive Maintenance Tabs ---
# Load and process data for all maintenance tabs
asset_ids = ['ESP_01', 'Pump_02', 'Valve_03', 'Compressor_04']
full_field_df = generate_full_field_data(asset_ids)
anomaly_model = train_anomaly_model(full_field_df)
df_with_anomalies = predict_anomalies(anomaly_model, full_field_df.copy())
active_alerts_df = df_with_anomalies[(df_with_anomalies['temperature'] > temp_threshold) | (df_with_anomalies['vibration'] > vib_threshold)]

with selected_tabs[4]: # Asset Health Overview
    st.header("Field-Wide Asset Health")
    st.markdown("Select an asset to view its historical temperature and vibration data. Any detected anomalies will be highlighted.")
    selected_asset = st.selectbox("Select an Asset to Inspect", asset_ids)
    
    asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset]
    
    # Create a chart with anomalies highlighted
    fig = px.line(asset_df, x='timestamp', y='temperature', title=f'{selected_asset} - Temperature Profile')
    anomalies = asset_df[asset_df['is_anomaly'] == 1]
    fig.add_trace(px.scatter(anomalies, x='timestamp', y='temperature').data[0].update(mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
    st.plotly_chart(fig, use_container_width=True)

with selected_tabs[5]: # AI Forecast & Anomalies
    st.header("AI-Powered Forecasts")
    st.markdown("This tab uses an LSTM neural network to forecast future equipment temperatures, allowing for proactive intervention.")
    selected_asset_forecast = st.selectbox("Select Asset for AI Forecasting", asset_ids, key="forecast_select")
    
    forecast_asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset_forecast]
    
    with st.spinner(f"Training LSTM model for {selected_asset_forecast}..."):
        lstm_model, scaler = train_lstm_model(forecast_asset_df, feature='temperature')
        forecasted_temp = predict_next_hour_temp(lstm_model, scaler, forecast_asset_df, feature='temperature')

    st.metric(label=f"Predicted Temperature in the Next Hour for {selected_asset_forecast}", value=f"{forecasted_temp:.2f}°F")
    if forecasted_temp > temp_threshold:
        st.warning("Forecasted temperature exceeds the safety threshold! Intervention may be required soon.")
    else:
        st.success("Forecasted temperature is within the normal operating range.")

with selected_tabs[6]: # Maintenance Plan
    st.header("Alerts & Optimized Maintenance Schedule")
    
    # Send email alert if needed
    if not active_alerts_df.empty and not st.session_state['summary_sent']:
        send_summary_email(active_alerts_df)
        st.session_state['summary_sent'] = True

    df_with_anomalies['downtime_risk'] = df_with_anomalies['is_anomaly'] * downtime_cost
    maintenance_schedule = df_with_anomalies.groupby('asset_id')['downtime_risk'].sum().sort_values(ascending=False).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚨 Active Critical Alerts")
        st.dataframe(active_alerts_df[['timestamp', 'asset_id', 'temperature', 'vibration']])
    with col2:
        st.subheader("📅 Prioritized Maintenance Plan")
        st.dataframe(maintenance_schedule.rename(columns={'downtime_risk': 'Estimated Downtime Cost ($)'}))

# --- REAL-TIME LOOP for Drilling Dashboard (at the end) ---
# This loop only affects the live drilling tab
while True:
    # ... code for updating live drilling dashboard ...
    time.sleep(1)
