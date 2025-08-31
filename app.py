import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Drilling Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Clean, Light Gradient Theme ---
st.markdown("""
<style>
    /* Main app background with a subtle gradient */
    .stApp {
        background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #31333F;
    }
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
    }
    /* Expander style in sidebar */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    /* Info box style in sidebar */
    [data-testid="stInfo"] {
        background-color: rgba(230, 247, 255, 0.8);
    }
    /* Metric label text */
    [data-testid="stMetricLabel"] {
        color: #5A5A5A;
    }
    /* Alert styles */
    [data-testid="stAlert"][data-baseweb="alert-success"] {
        background-color: rgba(45, 158, 87, 0.1);
        border: 1px solid rgba(45, 158, 87, 0.5);
    }
    [data-testid="stAlert"][data-baseweb="alert-warning"] {
        background-color: rgba(225, 179, 42, 0.1);
        border: 1px solid rgba(225, 179, 42, 0.5);
    }
    /* Title and header colors */
    h1, h2, h3 {
        color: #0E1117;
    }
    /* Custom style for the summary page for better text presentation */
    .summary-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'drilling_data' not in st.session_state:
    st.session_state.drilling_data = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])
if 'session_history' not in st.session_state:
    st.session_state.session_history = pd.DataFrame(columns=["Timestamp", "RPM", "Torque", "Vibration", "Bit Wear Index", "ROP (ft/hr)"])

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
tab1, tab2, tab3 = st.tabs(["**Executive Summary**", "**📊 Live Dashboard**", "**📈 Performance Overview**"])

# --- Tab 1: Executive Summary ---
with tab1:
    st.header("Real-Time Drilling Performance Dashboard")
    st.markdown('<p class="summary-text">Welcome to the LIVE Predictive Maintenance & Drilling Dashboard, a sophisticated monitoring tool designed specifically for the complexities of modern oil and gas drilling operations.</p>', unsafe_allow_html=True)
    st.markdown('<p class="summary-text">Developed by <strong>Mr. Omar Nur, a Petroleum Engineer</strong> with deep domain expertise, this application translates raw sensor data into actionable intelligence, empowering drilling teams to optimize performance and enhance operational safety.</p>', unsafe_allow_html=True)
    
    st.subheader("Application in Drilling Operations")
    st.markdown("""
    <div class="summary-text">
    This tool provides critical insights for multiple roles on the rig site and in the office:
    <ul>
        <li><strong>For Drilling Engineers & Supervisors:</strong> Monitor real-time Rate of Penetration (ROP), Torque, and RPM to make immediate adjustments to drilling parameters. The predictive alerts for high vibration can preempt equipment failure, significantly reducing non-productive time (NPT).</li>
        <li><strong>For Maintenance Teams:</strong> Utilize the Bit Wear Index and vibration trends to shift from reactive to predictive maintenance. By identifying potential issues before they become critical, teams can plan interventions, reduce costs, and improve rig safety.</li>
        <li><strong>For Data Analysts & Management:</strong> The session-wide KPIs on the 'Performance Overview' tab provide invaluable data for post-well analysis, benchmarking performance across different operations, and identifying long-term optimization strategies.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("The Result: An Intelligent Drilling Operation")
    st.markdown("""
    <div class="summary-text">
    The ultimate goal of this dashboard is to drive data-centric decision-making. By providing a clear, intuitive, and real-time view of downhole dynamics, this tool enables drilling teams to:
    <ul>
        <li>✅ <strong>Boost Efficiency:</strong> Achieve optimal ROP by balancing aggression with equipment stability.</li>
        <li>✅ <strong>Reduce Costs:</strong> Minimize NPT through early problem detection and proactive maintenance.</li>
        <li>✅ <strong>Enhance Safety:</strong> Avoid catastrophic equipment failures by monitoring vibration and torque anomalies.</li>
    </ul>
    This dashboard is more than a visualization tool; it is a step towards a more intelligent, efficient, and safer future for drilling operations in the oil and gas industry.
    </div>
    """, unsafe_allow_html=True)

# --- Tab 2: Live Dashboard ---
with tab2:
    st.subheader("Real-Time Sensor Data & Alerts")
    placeholder = st.empty()

# --- Tab 3: Performance Overview ---
with tab3:
    st.subheader("Session Performance Summary")
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

    new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque, "Vibration": new_vibration, "Bit Wear Index": new_bit_wear, "ROP (ft/hr)": new_rop}])

    # --- Update DataFrames in Session State ---
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
    st.session_state.session_history = pd.concat([st.session_state.session_history, new_row], ignore_index=True)

    # --- Update Tab 2: Live Dashboard ---
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
    
    # --- Update Tab 3: Performance Overview ---
    with overview_placeholder.container():
        history = st.session_state.session_history
        st.markdown("### Key Performance Indicators (Session Average)")
        avg_rop = history['ROP (ft/hr)'].mean()
        avg_torque = history['Torque'].mean()
        max_vibration = history['Vibration'].max()
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Average ROP (ft/hr)", f"{avg_rop:.2f}")
        kpi_col2.metric("Average Torque (Nm)", f"{avg_torque:.2f}")
        kpi_col3.metric("Max Vibration Recorded", f"{max_vibration:.2f}")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Operational Status")
        col1, col2 = st.columns([1,2])
        with col1:
            high_vibration_count = history[history['Vibration'] > vibration_threshold].shape[0]
            normal_count = history.shape[0] - high_vibration_count
            status_data = pd.DataFrame({'Status': ['Normal', 'High Vibration Alert'], 'Count': [normal_count, high_vibration_count]})
            fig = px.pie(status_data, values='Count', names='Status', title='Time in Operational Status', color='Status', color_discrete_map={'Normal':'#2E8B57', 'High Vibration Alert':'#FF8C00'})
            fig.update_layout(legend_title_text='Status', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Dashboard Interpretation")
            st.markdown("""
            This overview provides a comprehensive analysis of the drilling session, focusing on efficiency and predictive maintenance.
            - **Rate of Penetration (ROP)**: The primary measure of drilling speed.
            - **Torque**: Indicates the rotational force. Abnormally high torque can signal formation changes or mechanical problems.
            - **Vibration**: The key indicator of inefficiency and potential equipment failure.
            - **Bit Wear Index**: Simulates bit degradation, which directly impacts ROP.
            """)
    
    # --- Wait before the next update ---
    time.sleep(1)
