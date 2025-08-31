import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import matplotlib.pyplot as plt

# Import ONLY the required, lighter ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION (do this first) ---
st.set_page_config(
    page_title="Drilling & Maintenance Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ALL HELPER FUNCTIONS ---

# --- Predictive Maintenance Functions ---
@st.cache_data
def generate_full_field_data(asset_ids):
    all_asset_data = []
    for asset_id in asset_ids:
        time_index = pd.date_range(start='2025-08-01', periods=200, freq='h')
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

# Replaced LSTM with a lightweight Linear Regression model for stability
@st.cache_resource
def train_forecasting_model(_data):
    df = _data.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
    
    model = LinearRegression()
    model.fit(df[['time_idx', 'hour']], df['temperature'])
    
    last_timestamp = df['timestamp'].iloc[-1]
    next_timestamp = last_timestamp + pd.Timedelta(hours=1)
    next_time_idx = (next_timestamp - df['timestamp'].min()).dt.total_seconds() / 3600
    next_hour = next_timestamp.hour
    
    prediction_df = pd.DataFrame({'time_idx': [next_time_idx], 'hour': [next_hour]})
    prediction = model.predict(prediction_df)
    return prediction[0]

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

# --- 3. SESSION STATE & ONE-TIME DATA PREPARATION ---
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])

if 'maintenance_data_ready' not in st.session_state:
    asset_ids = ['ESP_01', 'Pump_02', 'Valve_03', 'Compressor_04']
    full_data = generate_full_field_data(asset_ids)
    st.session_state.full_field_df = full_data
    
    anomaly_model = train_anomaly_model(full_data)
    features = full_data[['temperature', 'vibration']]
    full_data['is_anomaly'] = anomaly_model.predict(features)
    full_data['is_anomaly'] = full_data['is_anomaly'].map({1: 0, -1: 1})
    st.session_state.df_with_anomalies = full_data
    
    st.session_state.maintenance_data_ready = True


# --- 4. UI AND APP LOGIC ---
st.title("Comprehensive Drilling & Asset Management Suite")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Drilling Suite Controls")
    rpm_mean = st.slider("Live Target RPM", 100, 200, 150)
    vibration_threshold = st.slider("Live Vibration Threshold", 0.8, 1.5, 1.0)
    st.header("Drillstring Simulator Controls")
    sim_rpm = st.slider("Simulated RPM", 30, 300, 120)
    sim_wob = st.slider("Simulated WOB (kN)", 10, 100, 50)
    sim_formation = st.selectbox("Formation Type", ["Soft", "Medium", "Hard"])
    sim_bit_type = st.selectbox("Bit Type", ["PDC", "Tricone", "Diamond"])
    sim_mud_type = st.selectbox("Mud Type", ["Water-based", "Oil-based", "Synthetic"])
    sim_bha_stiffness = st.slider("BHA Stiffness (kN/m)", 10, 100, 50)
    st.header("Predictive Maintenance Settings")
    temp_threshold = st.slider("Temperature Threshold (°F)", 80, 120, 95)
    vib_threshold = st.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)
    downtime_cost = st.slider("Downtime Cost per Hour ($)", 500, 5000, 1000, step=100)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Summary**", "**User Guide**", "**Live Drilling**", "**Simulator**", "**Asset Health**", "**AI Forecast**", "**Maintenance Plan**"])

with tab1:
    st.header("An Integrated Solution for Modern Oilfield Operations")
    st.markdown('<p class="guide-text">Developed by <strong>Mr. Omar Nur, a Petroleum Engineer</strong>, this application suite provides a holistic view of oilfield management, from real-time drilling optimization to full-field predictive asset maintenance.</p>', unsafe_allow_html=True)
    st.subheader("Suite Components")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Drilling Operations Suite")
        st.markdown("- **Live Dashboard:** Real-time monitoring of critical drilling parameters like ROP, Torque, and Vibration.\n- **Drilling Simulator:** A predictive tool to test and optimize drilling parameters *before* implementation, mitigating risks like stick-slip.")
    with col2:
        st.markdown("#### Predictive Maintenance Suite")
        st.markdown("- **Asset Health Overview:** Monitor the status of all field assets (ESPs, pumps, valves) from a central hub.\n- **AI Forecasting:** A lightweight model predicts equipment failures before they happen.\n- **Optimized Scheduling:** Data-driven maintenance plans prioritize work based on operational risk and cost.")
    st.markdown('<p class="guide-text">This integrated approach allows teams to move from a reactive to a proactive operational strategy, significantly reducing non-productive time (NPT), lowering maintenance costs, and enhancing overall safety.</p>', unsafe_allow_html=True)

with tab2:
    st.header("How to Use This Suite: A Practical Guide")
    st.subheader("Scenario 1: Using the Live Dashboard for Real-Time Adjustments")
    with st.expander("Click to see a Live Dashboard example"):
        st.markdown("""...""") # Add full user guide text
    st.subheader("Scenario 2: Using the Simulator for Proactive Planning")
    with st.expander("Click to see a Drillstring Simulator example"):
        st.markdown("""...""") # Add full user guide text

# --- Predictive Maintenance Tabs (using pre-loaded data) ---
df_with_anomalies = st.session_state.df_with_anomalies

with tab5:
    st.header("Field-Wide Asset Health")
    selected_asset = st.selectbox("Select an Asset to Inspect", df_with_anomalies['asset_id'].unique())
    asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset]
    fig = px.line(asset_df, x='timestamp', y='temperature', title=f'{selected_asset} - Temperature Profile')
    anomalies = asset_df[asset_df['is_anomaly'] == 1]
    if not anomalies.empty:
        fig.add_trace(px.scatter(anomalies, x='timestamp', y='temperature').data[0].update(mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("AI-Powered Forecasts")
    selected_asset_forecast = st.selectbox("Select Asset for AI Forecasting", df_with_anomalies['asset_id'].unique(), key="forecast_select")
    forecast_asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset_forecast]
    with st.spinner(f"Running forecast for {selected_asset_forecast}..."):
        forecasted_temp = train_forecasting_model(forecast_asset_df)
    st.metric(label=f"Predicted Temperature in the Next Hour for {selected_asset_forecast}", value=f"{forecasted_temp:.2f}°F")
    if forecasted_temp > temp_threshold:
        st.warning("Forecasted temperature exceeds the safety threshold!")
    else:
        st.success("Forecasted temperature is within the normal operating range.")

with tab7:
    st.header("Alerts & Optimized Maintenance Schedule")
    active_alerts_df = df_with_anomalies[(df_with_anomalies['temperature'] > temp_threshold) | (df_with_anomalies['vibration'] > vib_threshold)]
    df_with_anomalies['downtime_risk'] = df_with_anomalies['is_anomaly'] * downtime_cost
    maintenance_schedule = df_with_anomalies.groupby('asset_id')['downtime_risk'].sum().sort_values(ascending=False).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active Critical Alerts")
        st.dataframe(active_alerts_df[['timestamp', 'asset_id', 'temperature', 'vibration']])
    with col2:
        st.subheader("Prioritized Maintenance Plan")
        st.dataframe(maintenance_schedule.rename(columns={'downtime_risk': 'Estimated Downtime Cost ($)'}))

with tab4:
    st.header("Drillstring Dynamics Dashboard")
    sim_params = {"rpm": sim_rpm, "wob": sim_wob, "formation": sim_formation, "bit_type": sim_bit_type, "mud_type": sim_mud_type, "bha_stiffness": sim_bha_stiffness}
    time_sim, torque, displacement, rpm_series = run_drillstring_model(sim_params)
    ss_index, vibration_severity = compute_metrics(torque, rpm_series)
    st.subheader("Simulation Results")
    col1, col2 = st.columns(2)
    col1.metric("Stick-Slip Index", ss_index, help="A measure of torsional instability.")
    col2.metric("Vibration Severity", vibration_severity, help="Indicates axial vibration.")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(time_sim, torque, label="Torque (Nm)", color="dodgerblue"); ax.plot(time_sim, displacement, label="Axial Displacement (m)", color="darkorange")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Response"); ax.set_title("Drillstring Response Over Time"); ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig)
    results_df = pd.DataFrame({"Time (s)": time_sim, "Torque (Nm)": torque, "Displacement (m)": displacement, "RPM": rpm_series})
    csv = results_df.to_csv(index=False); st.download_button(label="Download Results as CSV", data=csv, file_name="drillstring_results.csv", mime="text/csv")

# --- LIVE DRILLING TAB (Must be last to allow st.rerun) ---
with tab3:
    st.header("Real-Time Drilling Monitor")
    placeholder = st.empty()
    
    new_timestamp = datetime.datetime.now()
    new_rpm = np.random.normal(loc=rpm_mean, scale=10)
    new_torque = np.random.normal(loc=500, scale=50)
    new_vibration = np.random.normal(loc=0.5, scale=0.1)
    new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque, "Vibration": new_vibration}])
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
    
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}")
        kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}")
        kpi3.metric(label="Live Vibration", value=f"{new_vibration:.2f}")
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp"))

    time.sleep(1)
    st.rerun()
