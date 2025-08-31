import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time # Import the time library for delays

# --- Page Configuration (set once at the top) ---
st.set_page_config(
    page_title="LIVE Drilling Dashboard", layout="wide", initial_sidebar_state="expanded"
)

# --- Sidebar (remains static and is always available) ---
with st.sidebar.expander("📘 Glossary of Drilling Terms"):
    st.markdown("- **BHA**: Bottom Hole Assembly\n- **MWD**: Measurement While Drilling\n- **ROP**: Rate of Penetration\n- **Bit Wear Index**: Metric for bit degradation")
st.sidebar.info("This dashboard simulates a live drilling operation with data updating every second.")

# --- Initialize Session State ---
# This is the key to making the app "remember" data between updates.
# We will store our growing DataFrame here.
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=[
        "Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"
    ])

# --- Main Dashboard Title (static) ---
st.title("🔴 LIVE Predictive Maintenance & Drilling Dashboard")
st.subheader("📊 Real-Time Sensor Data & Alerts")

# --- Interactive Controls (static) ---
# These controls will influence the data being generated in real-time.
rpm_mean = st.slider("Set Target RPM", 100, 200, 150)
vibration_threshold = st.slider("Set Vibration Alert Threshold", 0.8, 1.5, 1.0)

# --- Create Placeholders ---
# We create empty containers that we can fill and replace in our loop.
placeholder = st.empty()

# --- THE REAL-TIME SIMULATION LOOP ---
while True:
    # --- Generate ONE new row of data ---
    # We simulate a single moment in time.
    new_timestamp = datetime.datetime.now()
    new_rpm = np.random.normal(loc=rpm_mean, scale=10)
    new_torque = np.random.normal(loc=500, scale=50)
    # Occasionally inject a vibration anomaly
    if np.random.rand() > 0.95:
        new_vibration = np.random.normal(loc=0.9, scale=0.2)
    else:
        new_vibration = np.random.normal(loc=0.5, scale=0.1)

    # Bit wear and ROP should evolve over time based on previous values
    last_wear = st.session_state.drilling_data["Bit Wear Index"].iloc[-1] if not st.session_state.drilling_data.empty else 0
    new_bit_wear = np.clip(last_wear + np.random.normal(0.0005, 0.0002), 0, 1)
    new_rop = np.clip(100 - new_bit_wear * 80 + np.random.normal(0, 2), 20, 100)

    # Create a DataFrame for the new row
    new_row = pd.DataFrame([{
        "Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque,
        "Vibration": new_vibration, "Bit Wear Index": new_bit_wear, "ROP (ft/hr)": new_rop
    }])

    # --- Update the main DataFrame in Session State ---
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True)
    # Keep only the last 200 data points to prevent the app from slowing down (a "sliding window")
    st.session_state.drilling_data = st.session_state.drilling_data.tail(200)

    # --- Fill the placeholder with the updated dashboard ---
    with placeholder.container():
        # Create three columns for metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}")
        kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}")
        kpi3.metric(label="Live ROP (ft/hr)", value=f"{new_rop:.1f}")

        # Display the live-updating charts
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque"]], use_container_width=True)
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Vibration"]], use_container_width=True)

        # Update alerts in real-time
        st.subheader("⚠️ System Alerts")
        if new_vibration > vibration_threshold:
            st.warning(f"HIGH VIBRATION DETECTED! Current Level: {new_vibration:.2f}")
        else:
            st.success(f"Vibration levels normal. Current Level: {new_vibration:.2f}")

        st.subheader("📈 Optimization Insights")
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Bit Wear Index", "ROP (ft/hr)"]], use_container_width=True)

    # --- Wait for one second before the next update ---
    time.sleep(1)
