import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Drilling & Maintenance Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HELPER FUNCTIONS (DEFINED ONCE) ---

# --- Predictive Maintenance Functions ---
@st.cache_data
def generate_full_field_data(asset_ids):
    # This is cached, so it only runs once
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

# --- Drillstring Simulator Functions ---
def run_drillstring_model(params):
    # This function is fast and doesn't need caching
    time_sim = np.linspace(0, 10, 1000)
    freq = params["rpm"] / 60
    # ... (rest of the function is the same)
    friction = {"Soft": 0.8, "Medium": 1.0, "Hard": 1.2}[params["formation"]]
    bit = {"PDC": 1.0, "Tricone": 0.9, "Diamond": 1.1}[params["bit_type"]]
    mud = {"Water-based": 1.0, "Oil-based": 1.05, "Synthetic": 1.1}[params["mud_type"]]
    stiffness = params["bha_stiffness"] / 50
    torque = params["wob"] * np.sin(2 * np.pi * freq * time_sim) * friction * bit * mud
    displacement = np.cumsum(torque) * 0.01 / stiffness
    return time_sim, torque, displacement

# --- 3. SESSION STATE INITIALIZATION ---
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
if 'maintenance_data_loaded' not in st.session_state:
    st.session_state.maintenance_data_loaded = False
if 'anomaly_model' not in st.session_state:
    st.session_state.anomaly_model = None
if 'lstm_models' not in st.session_state:
    st.session_state.lstm_models = {}

# --- 4. UI AND APP LOGIC ---
st.title("Comprehensive Drilling & Asset Management Suite")

# --- SIDEBAR (Always Visible) ---
with st.sidebar:
    st.header("Drilling Suite Controls")
    rpm_mean = st.slider("Live Target RPM", 100, 200, 150)
    vibration_threshold = st.slider("Live Vibration Threshold", 0.8, 1.5, 1.0)
    st.header("Drillstring Simulator Controls")
    sim_rpm = st.slider("Simulated RPM", 30, 300, 120)
    sim_wob = st.slider("Simulated WOB (kN)", 10, 100, 50)
    # ... other sidebar sliders ...
    st.header("Predictive Maintenance Settings")
    temp_threshold = st.slider("Temperature Threshold (°F)", 80, 120, 95)
    vib_threshold = st.slider("Vibration Threshold (g)", 0.2, 1.0, 0.5, step=0.05)


# --- TABS ---
tab_names = ["Executive Summary", "User Guide", "Live Drilling", "Drilling Simulator", "Asset Health", "AI Forecast", "Maintenance Plan"]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([f"**{name}**" for name in tab_names])

# --- STATIC TABS (Load Instantly) ---
with tab1:
    st.header("An Integrated Solution for Modern Oilfield Operations")
    # ... full text content ...

with tab2:
    st.header("How to Use This Suite: A Practical Guide")
    # ... full text content ...

with tab4:
    st.header("Drillstring Dynamics Dashboard")
    sim_params = {"rpm": sim_rpm, "wob": sim_wob, "formation": "Medium", "bit_type": "PDC", "mud_type": "Water-based", "bha_stiffness": 50}
    time_sim, torque, displacement = run_drillstring_model(sim_params)
    fig, ax = plt.subplots(); ax.plot(time_sim, torque, label="Torque"); ax.plot(time_sim, displacement, label="Displacement"); ax.legend(); st.pyplot(fig)

# --- DYNAMIC & HEAVY TABS (Lazy Loading) ---
# Prepare data for all maintenance tabs
asset_ids = ['ESP_01', 'Pump_02', 'Valve_03', 'Compressor_04']
if st.session_state.maintenance_data_loaded:
    full_field_df = st.session_state.full_field_df
    df_with_anomalies = st.session_state.df_with_anomalies
else:
    # This block will run when a user first clicks a maintenance tab
    # We use a placeholder to show the loading status
    placeholder = st.empty()
    with placeholder.container():
        with st.spinner("Loading asset data and training AI models for the first time..."):
            # Import heavy libraries ONLY when needed
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Input
            
            # 1. Generate Data
            st.session_state.full_field_df = generate_full_field_data(asset_ids)
            full_field_df = st.session_state.full_field_df
            
            # 2. Train Anomaly Model
            anomaly_model = IsolationForest(contamination=0.05, random_state=42)
            anomaly_model.fit(full_field_df[['temperature', 'vibration']])
            st.session_state.anomaly_model = anomaly_model
            
            # 3. Predict Anomalies
            features = full_field_df[['temperature', 'vibration']]
            full_field_df['is_anomaly'] = anomaly_model.predict(features)
            full_field_df['is_anomaly'] = full_field_df['is_anomaly'].map({1: 0, -1: 1})
            st.session_state.df_with_anomalies = full_field_df
            df_with_anomalies = st.session_state.df_with_anomalies
            
            # 4. Set flag to true
            st.session_state.maintenance_data_loaded = True
    placeholder.empty() # Remove the loading message

with tab5:
    st.header("Field-Wide Asset Health")
    if st.session_state.maintenance_data_loaded:
        selected_asset = st.selectbox("Select an Asset to Inspect", asset_ids)
        asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset]
        fig = px.line(asset_df, x='timestamp', y='temperature', title=f'{selected_asset} - Temperature Profile')
        anomalies = asset_df[asset_df['is_anomaly'] == 1]
        fig.add_trace(px.scatter(anomalies, x='timestamp', y='temperature').data[0].update(mode='markers', marker=dict(color='red', size=8)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AI Models are loading. Please wait a moment...")

with tab6:
    st.header("AI-Powered Forecasts")
    if st.session_state.maintenance_data_loaded:
        # Lazy load LSTM model for the selected asset
        selected_asset_forecast = st.selectbox("Select Asset for AI Forecasting", asset_ids, key="forecast_select")
        if selected_asset_forecast not in st.session_state.lstm_models:
            with st.spinner(f"Training LSTM model for {selected_asset_forecast} for the first time..."):
                from sklearn.preprocessing import MinMaxScaler
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Input
                
                forecast_asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset_forecast]
                scaler = MinMaxScaler()
                scaled_series = scaler.fit_transform(forecast_asset_df[['temperature']])
                X_train, y_train = [], []
                for i in range(len(scaled_series) - 10):
                    X_train.append(scaled_series[i:i+10]); y_train.append(scaled_series[i+10])
                X_train, y_train = np.array(X_train), np.array(y_train)
                
                lstm_model = Sequential([Input(shape=(10, 1)), LSTM(50, activation='relu'), Dense(1)])
                lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                lstm_model.fit(X_train, y_train, epochs=1, verbose=0)
                st.session_state.lstm_models[selected_asset_forecast] = (lstm_model, scaler)

        lstm_model, scaler = st.session_state.lstm_models[selected_asset_forecast]
        forecast_asset_df = df_with_anomalies[df_with_anomalies['asset_id'] == selected_asset_forecast]
        scaled_series = scaler.transform(forecast_asset_df[['temperature']])
        last_sequence = scaled_series[-10:].reshape(1, 10, 1)
        predicted_value = lstm_model.predict(last_sequence, verbose=0)
        forecasted_temp = scaler.inverse_transform(predicted_value)[0][0]
        
        st.metric(label=f"Predicted Temperature in the Next Hour", value=f"{forecasted_temp:.2f}°F")
    else:
        st.info("AI Models are loading. Please wait a moment...")

with tab7:
    st.header("Alerts & Optimized Maintenance Schedule")
    if st.session_state.maintenance_data_loaded:
        active_alerts_df = df_with_anomalies[(df_with_anomalies['temperature'] > temp_threshold) | (df_with_anomalies['vibration'] > vib_threshold)]
        st.dataframe(active_alerts_df)
    else:
        st.info("AI Models are loading. Please wait a moment...")

# --- LIVE DRILLING TAB (at the very end) ---
with tab3:
    st.header("Real-Time Drilling Monitor")
    placeholder = st.empty()
    
    new_timestamp = datetime.datetime.now()
    # ... generate new row data ...
    new_rpm = np.random.normal(loc=rpm_mean, scale=10)
    new_torque = np.random.normal(loc=500, scale=50)
    new_vibration = np.random.normal(loc=0.9, scale=0.2) if np.random.rand() > 0.95 else np.random.normal(loc=0.5, scale=0.1)
    new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque, "Vibration": new_vibration}])
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
    
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}")
        kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}")
        # Add other metrics if needed
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque", "Vibration"]])

    time.sleep(1)
    st.rerun()
