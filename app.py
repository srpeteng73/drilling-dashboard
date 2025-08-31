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
    page_title="Drilling & Maintenance Suite by Omar Nur",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HELPER FUNCTIONS ---

@st.cache_data
def generate_full_field_data(asset_ids):
    """Generates a smaller, memory-efficient dataset."""
    all_asset_data = []
    # REDUCED DATA SIZE for stability on free hosting
    num_periods = 100 
    for asset_id in asset_ids:
        time_index = pd.date_range(start='2025-01-01', periods=num_periods, freq='h')
        temp = np.random.normal(loc=75, scale=2, size=num_periods)
        vibe = np.random.normal(loc=0.3, scale=0.05, size=num_periods)
        anomaly_start = np.random.randint(50, 80)
        temp[anomaly_start:anomaly_start+10] += np.linspace(5, 15, 10)
        vibe[anomaly_start:anomaly_start+10] += np.linspace(0.1, 0.4, 10)
        all_asset_data.append(pd.DataFrame({'timestamp': time_index, 'asset_id': asset_id, 'temperature': temp, 'vibration': vibe}))
    return pd.concat(all_asset_data, ignore_index=True)

@st.cache_resource
def train_anomaly_model(_data):
    """Trains the anomaly detection model. Cached to run only once."""
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(_data[['temperature', 'vibration']])
    return model

@st.cache_resource
def train_forecasting_model(_data):
    """Trains a lightweight forecasting model. Cached per asset."""
    df = _data.copy()
    df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
    model = LinearRegression()
    model.fit(df[['time_idx']], df['temperature'])
    last_time_idx = df['time_idx'].iloc[-1]
    prediction = model.predict(np.array([[last_time_idx + 1]]))
    return prediction[0]

def run_drillstring_model(params):
    time_sim = np.linspace(0, 10, 500) # Reduced points for faster plotting
    freq = params["rpm"] / 60
    friction = {"Soft": 0.8, "Medium": 1.0, "Hard": 1.2}[params["formation"]]
    bit = {"PDC": 1.0, "Tricone": 0.9, "Diamond": 1.1}[params["bit_type"]]
    torque = params["wob"] * 0.05 * np.sin(2 * np.pi * freq * time_sim) * friction * bit
    displacement = np.cumsum(torque) * 0.01
    rpm_series = np.full_like(time_sim, params["rpm"]) + np.random.normal(0, 1, size=time_sim.shape)
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
    st.markdown("---")
    st.header("Predictive Maintenance Settings")
    temp_threshold = st.slider("Temperature Threshold (°F)", 80, 120, 95)
    vib_threshold = st.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)
    downtime_cost = st.slider("Downtime Cost per Hour ($)", 500, 5000, 1000, step=100)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Summary**", "**User Guide**", "**Live Drilling**", "**Simulator**", "**Asset Health**", "**AI Forecast**", "**Maintenance Plan**"])

with tab1:
    st.header("An Integrated Solution for Modern Oilfield Operations")
    st.markdown('Developed by **Mr. Omar Nur, a Petroleum Engineer**, this application suite provides a holistic view of oilfield management, from real-time drilling optimization to full-field predictive asset maintenance.')
    # ... (Full Summary Content)

with tab2:
    st.header("How to Use This Suite: A Practical Guide")
    # ... (Full User Guide Content)

with tab3:
    st.header("Real-Time Drilling Monitor")
    if st.button("Refresh Live Data"):
        new_row = pd.DataFrame([{
            "Timestamp": datetime.datetime.now(),
            "RPM": np.random.normal(loc=rpm_mean, scale=10),
            "Torque": np.random.normal(loc=500, scale=50),
            "Vibration": np.random.normal(loc=0.5, scale=0.1),
            "Bit Wear Index": 0, "ROP (ft/hr)": 0 # Simplified for stability
        }])
        st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(100)
        st.toast("Live data updated!")

    st.subheader("Current Drilling Parameters")
    if not st.session_state.drilling_data.empty:
        latest_data = st.session_state.drilling_data.iloc[-1]
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Live RPM", f"{latest_data['RPM']:.1f}")
        kpi2.metric("Live Torque (Nm)", f"{latest_data['Torque']:.1f}")
        kpi3.metric("Live Vibration", f"{latest_data['Vibration']:.2f}")
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque", "Vibration"]])
    else:
        st.info("Click 'Refresh Live Data' to start monitoring.")

with tab4:
    st.header("Drillstring Dynamics Dashboard")
    sim_params = {"rpm": sim_rpm, "wob": sim_wob, "formation": sim_formation, "bit_type": sim_bit_type}
    time_sim, torque, displacement, rpm_series = run_drillstring_model(sim_params)
    ss_index, vibration_severity = compute_metrics(torque, rpm_series)
    col1, col2 = st.columns(2)
    col1.metric("Stick-Slip Index", f"{ss_index:.2f}")
    col2.metric("Vibration Severity", f"{vibration_severity:.2f}")
    fig, ax = plt.subplots(); ax.plot(time_sim, torque, label="Torque"); ax.plot(time_sim, displacement, label="Displacement"); ax.legend(); st.pyplot(fig)

# --- Predictive Maintenance Tabs with On-Demand Loading ---
if not st.session_state.maintenance_data_ready:
    st.info("The Predictive Maintenance module is not loaded to ensure a fast startup. Click the button below to activate it.")
    if st.button("Load Asset Data & Train AI Models"):
        with st.spinner("Performing one-time data generation and model training... Please wait."):
            asset_ids = ['ESP_01', 'Pump_02', 'Valve_03'] # Reduced assets for memory
            full_data = generate_full_field_data(asset_ids)
            st.session_state.full_field_df = full_data
            anomaly_model = train_anomaly_model(full_data)
            full_data['is_anomaly'] = anomaly_model.predict(full_data[['temperature', 'vibration']])
            full_data['is_anomaly'] = full_data['is_anomaly'].map({1: 0, -1: 1})
            st.session_state.df_with_anomalies = full_data
            st.session_state.maintenance_data_ready = True
        st.success("Maintenance module loaded successfully!")
        st.rerun()
else:
    df_with_anomalies = st.session_state.df_with_anomalies
    with tab5:
        st.header("Field-Wide Asset Health")
        selected_asset = st.selectbox("Select an Asset", df_with_anomalies['asset_id'].unique())
        asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset]
        fig = px.line(asset_df, x='timestamp', y='temperature', title=f'Temperature Profile')
        anomalies = asset_df[asset_df['is_anomaly'] == 1]
        if not anomalies.empty:
            fig.add_trace(px.scatter(anomalies, x='timestamp', y='temperature').data[0].update(mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.header("AI-Powered Forecasts")
        selected_asset_forecast = st.selectbox("Select Asset for Forecasting", df_with_anomalies['asset_id'].unique())
        forecast_asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset_forecast]
        with st.spinner(f"Running forecast..."):
            forecasted_temp = train_forecasting_model(forecast_asset_df)
        st.metric(f"Predicted Temperature in Next Hour", f"{forecasted_temp:.2f}°F")

    with tab7:
        st.header("Alerts & Optimized Maintenance Schedule")
        active_alerts = df_with_anomalies[(df_with_anomalies['temperature'] > temp_threshold) | (df_with_anomalies['vibration'] > vib_threshold)]
        st.subheader("Active Critical Alerts")
        st.dataframe(active_alerts)
