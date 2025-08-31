import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Drilling & Maintenance Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HELPER FUNCTIONS ---

# These functions are only called AFTER the user clicks a button
@st.cache_data
def generate_full_field_data(asset_ids):
    all_asset_data = []
    for asset_id in asset_ids:
        time_index = pd.date_range(start='2025-01-01', periods=200, freq='h')
        temp = np.random.normal(loc=75, scale=2, size=200)
        vibe = np.random.normal(loc=0.3, scale=0.05, size=200)
        anomaly_start = np.random.randint(100, 180)
        temp[anomaly_start:anomaly_start+10] += np.linspace(5, 25, 10)
        vibe[anomaly_start:anomaly_start+10] += np.linspace(0.1, 0.6, 10)
        all_asset_data.append(pd.DataFrame({'timestamp': time_index, 'asset_id': asset_id, 'temperature': temp, 'vibration': vibe}))
    return pd.concat(all_asset_data, ignore_index=True)

@st.cache_resource
def train_anomaly_model(_data):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(_data[['temperature', 'vibration']])
    return model

@st.cache_resource
def train_forecasting_model(_data):
    df = _data.copy()
    df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
    model = LinearRegression()
    model.fit(df[['time_idx']], df['temperature'])
    last_time_idx = df['time_idx'].iloc[-1]
    prediction = model.predict(np.array([[last_time_idx + 1]]))
    return prediction[0]

def run_drillstring_model(params):
    time_sim = np.linspace(0, 10, 1000)
    freq = params["rpm"] / 60
    friction = {"Soft": 0.8, "Medium": 1.0, "Hard": 1.2}[params["formation"]]
    bit = {"PDC": 1.0, "Tricone": 0.9, "Diamond": 1.1}[params["bit_type"]]
    mud = {"Water-based": 1.0, "Oil-based": 1.05, "Synthetic": 1.1}[params["mud_type"]]
    stiffness = params["bha_stiffness"] / 50
    torque = params["wob"] * 0.05 * np.sin(2 * np.pi * freq * time_sim) * friction * bit * mud
    displacement = np.cumsum(torque) * 0.01 / stiffness
    rpm_series = np.full_like(time_sim, params["rpm"]) + np.random.normal(0, 2, size=time_sim.shape)
    return time_sim, torque, displacement, rpm_series

def compute_metrics(torque, rpm_series):
    ss_index = np.std(torque) / max(np.mean(torque), 1e-6)
    vibration_severity = np.max(np.abs(np.diff(rpm_series)))
    return round(ss_index, 3), round(vibration_severity, 3)

# --- 3. UI and APP LOGIC ---
st.title("Comprehensive Drilling & Asset Management Suite")

# --- Initialize Session State ---
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
if 'maintenance_data_ready' not in st.session_state:
    st.session_state.maintenance_data_ready = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("Drilling Suite Controls")
    rpm_mean = st.slider("Live Target RPM", 100, 200, 150)
    vibration_threshold = st.slider("Live Vibration Threshold", 0.8, 1.5, 1.0)
    st.markdown("---")
    st.header("Drillstring Simulator Controls")
    sim_rpm = st.slider("Simulated RPM", 30, 300, 120)
    sim_wob = st.slider("Simulated WOB (kN)", 10, 100, 50)
    sim_formation = st.selectbox("Formation Type", ["Soft", "Medium", "Hard"])
    sim_bit_type = st.selectbox("Bit Type", ["PDC", "Tricone", "Diamond"])
    sim_mud_type = st.selectbox("Mud Type", ["Water-based", "Oil-based", "Synthetic"])
    sim_bha_stiffness = st.slider("BHA Stiffness (kN/m)", 10, 100, 50)
    st.markdown("---")
    st.header("Predictive Maintenance Settings")
    temp_threshold = st.slider("Temperature Threshold (°F)", 80, 120, 95)
    vib_threshold = st.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)
    downtime_cost = st.slider("Downtime Cost per Hour ($)", 500, 5000, 1000, step=100)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Summary**", "**User Guide**", "**Live Drilling**", "**Simulator**", "**Asset Health**", "**AI Forecast**", "**Maintenance Plan**"])

with tab1:
    st.header("An Integrated Solution for Modern Oilfield Operations")
    # ... (Full Summary Content)

with tab2:
    st.header("How to Use This Suite: A Practical Guide")
    # ... (Full User Guide Content)

with tab3:
    st.header("Real-Time Drilling Monitor")
    if st.button("Refresh Live Data"):
        new_timestamp = datetime.datetime.now()
        new_rpm = np.random.normal(loc=rpm_mean, scale=10)
        # ... (Full data generation logic)
        new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, ...}])
        st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
        st.toast("Live data updated!")

    st.subheader("Current Drilling Parameters")
    if not st.session_state.drilling_data.empty:
        # ... (Full display logic for KPIs, Alerts, and Charts)
    else:
        st.info("Click 'Refresh Live Data' to start monitoring.")

with tab4:
    st.header("Drillstring Dynamics Dashboard")
    # ... (Full Simulator Content)

# --- Predictive Maintenance Tabs with On-Demand Loading ---
placeholder = st.container() # A single placeholder for all maintenance tabs

if not st.session_state.maintenance_data_ready:
    with placeholder:
        st.warning("The Predictive Maintenance module is not loaded.")
        if st.button("Load Asset Data & Train AI Models"):
            with st.spinner("Performing one-time data generation and model training... Please wait."):
                asset_ids = ['ESP_01', 'Pump_02', 'Valve_03', 'Compressor_04']
                full_data = generate_full_field_data(asset_ids)
                st.session_state.full_field_df = full_data
                
                anomaly_model = train_anomaly_model(full_data)
                features = full_data[['temperature', 'vibration']]
                full_data['is_anomaly'] = anomaly_model.predict(features)
                full_data['is_anomaly'] = full_data['is_anomaly'].map({1: 0, -1: 1})
                st.session_state.df_with_anomalies = full_data
                st.session_state.maintenance_data_ready = True
            st.rerun() # Rerun the script to now show the loaded tabs
else:
    df_with_anomalies = st.session_state.df_with_anomalies
    with tab5:
        st.header("Field-Wide Asset Health")
        selected_asset = st.selectbox("Select an Asset to Inspect", df_with_anomalies['asset_id'].unique())
        # ... (Full Asset Health display logic)

    with tab6:
        st.header("AI-Powered Forecasts")
        selected_asset_forecast = st.selectbox("Select Asset for AI Forecasting", df_with_anomalies['asset_id'].unique(), key="forecast_select")
        # ... (Full AI Forecast display logic)

    with tab7:
        st.header("Alerts & Optimized Maintenance Schedule")
        # ... (Full Maintenance Plan display logic)
