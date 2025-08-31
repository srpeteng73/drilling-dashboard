import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="LIVE Drilling Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Clean, Light Gradient Theme ---
st.markdown("""
<style>
    /* Main app background with a subtle gradient */
    .stApp {
        background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #31333F; /* Darker text for readability */
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white */
        backdrop-filter: blur(5px);
    }

    /* Style for the expander in the sidebar */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    
    /* Style for the info box in the sidebar */
    [data-testid="stInfo"] {
        background-color: rgba(230, 247, 255, 0.8);
    }

    /* Metric label text (e.g., "Live RPM") */
    [data-testid="stMetricLabel"] {
        color: #5A5A5A;
    }

    /* Style for success alerts (green) */
    [data-testid="stAlert"][data-baseweb="alert-success"] {
        background-color: rgba(45, 158, 87, 0.1);
        border: 1px solid rgba(45, 158, 87, 0.5);
    }

    /* Style for warning alerts (yellow) */
    [data-testid="stAlert"][data-baseweb="alert-warning"] {
        background-color: rgba(225, 179, 42, 0.1);
        border: 1px solid rgba(225, 179, 42, 0.5);
    }

    /* Title and header colors */
    h1, h2, h3 {
        color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
# 'drilling_data' for the real-time sliding window chart
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=[
        "Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"
    ])
# 'session_history' to store all data for the overview tab
if 'session_history' not in st.session_state:
    st.session_state.session_history = pd.DataFrame(columns=[
        "Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"
    ])

# --- Sidebar Content ---
with st.sidebar.expander("📘 Glossary of Drilling Terms", expanded=True):
    st.markdown("- **BHA**: Bottom Hole Assembly\n- **MWD**: Measurement While Drilling\n- **ROP**: Rate of Penetration\n- **Bit Wear Index**: Metric for bit degradation")
st.sidebar.info("This dashboard simulates a live drilling operation with data updating every second.")

# --- Main Dashboard Title ---
st.title("🔴 LIVE Predictive Maintenance & Drilling Dashboard")

# --- Interactive Controls ---
rpm_mean = st.slider("Set Target RPM", 100, 200, 150)
vibration_threshold = st.slider("Set Vibration Alert Threshold", 0.8, 1.5, 1.0)

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📊 Live Dashboard", "📈 Performance Overview"])

# --- Tab 1: Live Dashboard ---
with tab1:
    st.subheader("Real-Time Sensor Data & Alerts")
    placeholder = st.empty()

# --- Tab 2: Performance Overview ---
with tab2:
    st.subheader("Session Performance Summary")
    
    # We will fill this tab with calculated KPIs and explanations
    overview_placeholder = st.empty()


# --- REAL-TIME SIMULATION LOOP ---
while True:
    # --- Generate New Data ---
    new_timestamp = datetime.datetime.now()
    new_rpm = np.random.normal(loc=rpm_mean, scale=10)
    new_torque = np.random.normal(loc=500, scale=50)
    if np.random.rand() > 0.95:
        new_vibration = np.random.normal(loc=0.9, scale=0.2)
    else:
        new_vibration = np.random.normal(loc=0.5, scale=0.1)

    last_wear = st.session_state.drilling_data["Bit Wear Index"].iloc[-1] if not st.session_state.drilling_data.empty else 0
    new_bit_wear = np.clip(last_wear + np.random.normal(0.0005, 0.0002), 0, 1)
    new_rop = np.clip(100 - new_bit_wear * 80 + np.random.normal(0, 2), 20, 100)

    new_row = pd.DataFrame([{
        "Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque,
        "Vibration": new_vibration, "Bit Wear Index": new_bit_wear, "ROP (ft/hr)": new_rop
    }])

    # --- Update DataFrames in Session State ---
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
    st.session_state.session_history = pd.concat([st.session_state.session_history, new_row], ignore_index=True)

    # --- Update Tab 1: Live Dashboard ---
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}")
        kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}")
        kpi3.metric(label="Live ROP (ft/hr)", value=f"{new_rop:.1f}")

        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque"]], use_container_width=True)
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Vibration"]], use_container_width=True)

        st.subheader("⚠️ System Alerts")
        if new_vibration > vibration_threshold:
            st.warning(f"HIGH VIBRATION DETECTED! Current Level: {new_vibration:.2f}")
        else:
            st.success(f"Vibration levels normal. Current Level: {new_vibration:.2f}")

        st.subheader("📈 Optimization Insights")
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Bit Wear Index", "ROP (ft/hr)"]], use_container_width=True)
    
    # --- Update Tab 2: Performance Overview ---
    with overview_placeholder.container():
        history = st.session_state.session_history
        
        st.markdown("### Key Performance Indicators (Session Average)")
        
        # Calculate KPIs
        avg_rpm = history['RPM'].mean()
        avg_torque = history['Torque'].mean()
        avg_rop = history['ROP (ft/hr)'].mean()
        max_vibration = history['Vibration'].max()
        current_bit_wear = history['Bit Wear Index'].iloc[-1]
        
        # Display KPIs
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Average ROP (ft/hr)", f"{avg_rop:.2f}")
        kpi_col2.metric("Average Torque (Nm)", f"{avg_torque:.2f}")
        kpi_col3.metric("Max Vibration Recorded", f"{max_vibration:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("### Operational Status")
        
        # Pie Chart for Operational Status
        col1, col2 = st.columns([1,2])
        with col1:
            high_vibration_count = history[history['Vibration'] > vibration_threshold].shape[0]
            normal_count = history.shape[0] - high_vibration_count
            
            status_data = pd.DataFrame({
                'Status': ['Normal', 'High Vibration Alert'],
                'Count': [normal_count, high_vibration_count]
            })
            
            fig = px.pie(status_data, values='Count', names='Status', title='Time in Operational Status',
                         color='Status', color_discrete_map={'Normal':'green', 'High Vibration Alert':'orange'})
            fig.update_layout(legend_title_text='Status', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Dashboard Interpretation")
            st.markdown("""
            This dashboard provides a comprehensive view of the drilling operation, focusing on efficiency and predictive maintenance.

            - **Rate of Penetration (ROP)**: This is the primary measure of drilling speed. Higher ROP is generally better, but must be balanced against equipment health.
            - **Torque**: Indicates the rotational force required to turn the drill bit. Abnormally high torque can signal formation changes or mechanical problems.
            - **Vibration**: Excessive vibration is a key indicator of inefficiency and potential equipment failure. The 'High Vibration Alert' status highlights periods where the machinery is under stress, allowing for proactive maintenance.
            - **Bit Wear Index**: This simulates the degradation of the drill bit. As the bit wears, ROP typically decreases, signaling that a bit change may be required for optimal performance.
            """)
    
    # --- Wait for one second before the next update ---
    time.sleep(1)
