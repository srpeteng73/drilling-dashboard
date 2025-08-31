import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import matplotlib.pyplot as plt
import os

# ML/AI Libraries - import them all at the top
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# --- 1. PAGE CONFIGURATION (do this first) ---
st.set_page_config(
    page_title="Drilling & Maintenance Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ALL HELPER FUNCTIONS ---

# --- Predictive Maintenance Functions ---
@st.cache_data
def generate_full_field_data(asset_ids, start_time='2025-08-01', total_periods=200):
    all_asset_data = []
    for asset_id in asset_ids:
        time_index = pd.date_range(start=start_time, periods=total_periods, freq='h')
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
    lstm_model = Sequential([Input(shape=(10, 1)), LSTM(50, activation='relu'), Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=1, verbose=0)
    return lstm_model, scaler

def predict_next_hour_temp(_model, _scaler, data, feature='temperature'):
    scaled_series = _scaler.transform(data[[feature]])
    last_sequence = scaled_series[-10:].reshape(1, 10, 1)
    predicted_value = _model.predict(last_sequence, verbose=0)
    return _scaler.inverse_transform(predicted_value)[0][0]

# --- Drillstring Simulator Functions ---
def run_drillstring_model(params):
    time_sim = np.linspace(0, 10, 1000)
    freq = params["rpm"] / 60
    friction = {"Soft": 0.8, "Medium": 1.0, "Hard": 1.2}[params["formation"]]
    bit = {"PDC": 1.0, "Tricone": 0.9, "Diamond": 1.1}[params["bit_type"]]
    mud = {"Water-based": 1.0, "Oil-based": 1.05, "Synthetic": 1.1}[params["mud_type"]]
    stiffness = params["bha_stiffness"] / 50
    torque = params["wob"] * np.sin(2 * np.pi * freq * time_sim) * friction * bit * mud
    displacement = np.cumsum(torque) * 0.01 / stiffness
    rpm_series = np.full_like(time_sim, params["rpm"]) + np.random.normal(0, 2, size=time_sim.shape)
    return time_sim, torque, displacement, rpm_series

def compute_metrics(torque, rpm_series):
    ss_index = np.std(torque) / max(np.mean(torque), 1e-6)
    vibration_severity = np.max(np.abs(np.diff(rpm_series)))
    return round(ss_index, 3), round(vibration_severity, 3)

# --- 3. MAIN APPLICATION FUNCTION ---
def main():
    # --- Custom CSS ---
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS hidden for brevity

    # --- SESSION STATE INITIALIZATION ---
    if 'drilling_data' not in st.session_state:
        st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
    if 'session_history' not in st.session_state:
        st.session_state.session_history = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])

    # --- SIDEBAR LAYOUT ---
    st.sidebar.title("Control Panel")
    st.sidebar.markdown("---")
    st.sidebar.header("Drilling Suite Controls")
    rpm_mean = st.sidebar.slider("Live Target RPM", 100, 200, 150)
    vibration_threshold = st.sidebar.slider("Live Vibration Threshold", 0.8, 1.5, 1.0)
    sim_rpm = st.sidebar.slider("Simulated RPM", 30, 300, 120)
    sim_wob = st.sidebar.slider("Simulated WOB (kN)", 10, 100, 50)
    sim_formation = st.sidebar.selectbox("Formation Type", ["Soft", "Medium", "Hard"])
    sim_bit_type = st.sidebar.selectbox("Bit Type", ["PDC", "Tricone", "Diamond"])
    sim_mud_type = st.sidebar.selectbox("Mud Type", ["Water-based", "Oil-based", "Synthetic"])
    sim_bha_stiffness = st.sidebar.slider("BHA Stiffness (kN/m)", 10, 100, 50)
    st.sidebar.markdown("---")
    st.sidebar.header("Predictive Maintenance Settings")
    temp_threshold = st.sidebar.slider("Temperature Threshold (°F)", 80, 120, 95)
    vib_threshold = st.sidebar.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)
    downtime_cost = st.sidebar.slider("Downtime Cost per Hour ($)", 500, 5000, 1000, step=100)

    # --- MAIN APPLICATION LAYOUT ---
    st.title("Comprehensive Drilling & Asset Management Suite")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Executive Summary**", "**User Guide**", "**Live Drilling**", "**Drilling Simulator**", "**Asset Health Overview**", "**AI Forecast & Anomalies**", "**Maintenance Plan**"])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab1:
        st.header("An Integrated Solution for Modern Oilfield Operations")
        # ... full text content ...

    # --- TAB 2: USER GUIDE ---
    with tab2:
        st.header("How to Use This Suite: A Practical Guide")
        # ... full text content ...
    
    # --- TAB 3: LIVE DRILLING (REFACTORED AND STABLE) ---
    with tab3:
        st.header("Real-Time Drilling Monitor")
        placeholder = st.empty()

        # Generate new data row
        new_timestamp = datetime.datetime.now()
        new_rpm = np.random.normal(loc=rpm_mean, scale=10)
        new_torque = np.random.normal(loc=500, scale=50)
        new_vibration = np.random.normal(loc=0.9, scale=0.2) if np.random.rand() > 0.95 else np.random.normal(loc=0.5, scale=0.1)
        last_wear = st.session_state.drilling_data["Bit Wear Index"].iloc[-1] if not st.session_state.drilling_data.empty else 0
        new_bit_wear = np.clip(last_wear + np.random.normal(0.0005, 0.0002), 0, 1)
        new_rop = np.clip(100 - new_bit_wear * 80 + np.random.normal(0, 2), 20, 100)
        
        new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque, "Vibration": new_vibration, "Bit Wear Index": new_bit_wear, "ROP (ft/hr)": new_rop}])
        
        # Update session state
        st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
        st.session_state.session_history = pd.concat([st.session_state.session_history, new_row], ignore_index=True)
        
        # Display content in placeholder
        with placeholder.container():
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}")
            kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}")
            kpi3.metric(label="Live ROP (ft/hr)", value=f"{new_rop:.1f}")
            st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque"]])
            st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Vibration"]])
            if new_vibration > vibration_threshold: st.warning(f"HIGH VIBRATION DETECTED! Level: {new_vibration:.2f}")
            else: st.success(f"Vibration levels normal. Level: {new_vibration:.2f}")

        # Auto-refresh mechanism
        time.sleep(1)
        st.rerun()

    # --- TAB 4: DRILLING SIMULATOR ---
    with tab4:
        st.header("Drillstring Dynamics Dashboard")
        sim_params = {"rpm": sim_rpm, "wob": sim_wob, "formation": sim_formation, "bit_type": sim_bit_type, "mud_type": sim_mud_type, "bha_stiffness": sim_bha_stiffness}
        time_sim, torque, displacement, rpm_series = run_drillstring_model(sim_params)
        # ... rest of the tab's code ...

    # --- PREDICTIVE MAINTENANCE TABS ---
    asset_ids = ['ESP_01', 'Pump_02', 'Valve_03', 'Compressor_04']
    full_field_df = generate_full_field_data(asset_ids)
    anomaly_model = train_anomaly_model(full_field_df)
    df_with_anomalies = predict_anomalies(anomaly_model, full_field_df.copy())

    with tab5:
        st.header("Field-Wide Asset Health")
        # ... rest of the tab's code ...

    with tab6:
        st.header("AI-Powered Forecasts")
        # ... rest of the tab's code ...

    with tab7:
        st.header("Alerts & Optimized Maintenance Schedule")
        # ... rest of the tab's code ...

# --- 4. RUN THE APP ---
if __name__ == "__main__":
    main()
