import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import matplotlib.pyplot as plt

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
    /* Custom style for text presentation */
    .guide-text {
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
st.sidebar.markdown("### Glossary & Info")
with st.sidebar.expander("📘 Glossary of Drilling Terms", expanded=False):
    st.markdown("- **BHA**: Bottom Hole Assembly\n- **MWD**: Measurement While Drilling\n- **ROP**: Rate of Penetration\n- **Bit Wear Index**: Metric for bit degradation")

st.sidebar.info("This application contains both a live drilling monitor and a dynamics simulator.")
st.sidebar.markdown("---")
st.sidebar.header("Live Dashboard Controls")
rpm_mean = st.sidebar.slider("Set Live Target RPM", 100, 200, 150)
vibration_threshold = st.sidebar.slider("Set Live Vibration Alert Threshold", 0.8, 1.5, 1.0)
st.sidebar.markdown("---")
st.sidebar.header("Drillstring Simulator Parameters")
sim_rpm = st.sidebar.slider("Rotary Speed (RPM)", min_value=30, max_value=300, value=120)
sim_wob = st.sidebar.slider("Weight on Bit (kN)", min_value=10, max_value=100, value=50)
sim_formation = st.sidebar.selectbox("Formation Type", ["Soft", "Medium", "Hard"])
sim_bit_type = st.sidebar.selectbox("Bit Type", ["PDC", "Tricone", "Diamond"])
sim_mud_type = st.sidebar.selectbox("Mud Type", ["Water-based", "Oil-based", "Synthetic"])
sim_bha_stiffness = st.sidebar.slider("BHA Stiffness (kN/m)", min_value=10, max_value=100, value=50)


# --- Main Application Layout ---
st.title("Comprehensive Drilling Performance & Simulation Suite")

# --- Create Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Executive Summary**", "**📖 User Guide & Examples**", "**📊 Live Dashboard**", "**📈 Performance Overview**", "**⚙️ Drillstring Simulator**"])

# --- Tab 1: Executive Summary ---
with tab1:
    st.header("Real-Time Drilling Performance Dashboard")
    st.markdown('<p class="guide-text">Developed by <strong>Mr. Omar Nur, a Petroleum Engineer</strong>, this application translates raw sensor data into actionable intelligence, empowering drilling teams to optimize performance and enhance operational safety.</p>', unsafe_allow_html=True)
    st.markdown('<p class="guide-text">The ultimate goal is to drive data-centric decision-making, enabling teams to boost efficiency, reduce costs, and enhance safety. This suite is a step towards a more intelligent future for drilling operations in the oil and gas industry.</p>', unsafe_allow_html=True)

# --- Tab 2: User Guide & Examples ---
with tab2:
    st.header("How to Use This Suite: A Practical Guide")
    
    st.subheader("Scenario 1: Using the Live Dashboard for Real-Time Adjustments")
    with st.expander("Click here to see a Live Dashboard example"):
        st.markdown("""
        <div class="guide-text">
        The <strong>Live Dashboard</strong> is your real-time eye on the operation. It helps you react instantly to changing downhole conditions.
        <br><br>
        <strong>Example Scenario:</strong>
        <ul>
            <li><strong>Situation:</strong> While drilling, you are monitoring the 'Live Dashboard' tab. You notice the <strong>Live ROP (ft/hr)</strong> metric begins to drop, and simultaneously, a yellow warning box appears: <strong>"HIGH VIBRATION DETECTED!"</strong>.</li>
            <li><strong>Analysis:</strong> This indicates that the current combination of parameters is likely causing resonant vibrations in the drillstring, which hinders performance and can cause equipment damage.</li>
            <li><strong>Action:</strong> In the sidebar under "Live Dashboard Controls," you slightly decrease the <strong>'Set Live Target RPM'</strong> slider from 150 to 140.</li>
            <li><strong>Result:</strong> You watch the 'Vibration' chart, and the erratic peaks immediately smooth out. The yellow warning box is replaced by a green 'Vibration levels normal' message. Shortly after, the ROP begins to recover. You have successfully identified and mitigated a harmful vibration issue in real-time.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Scenario 2: Using the Simulator for Proactive Planning")
    with st.expander("Click here to see a Drillstring Simulator example"):
        st.markdown("""
        <div class="guide-text">
        The <strong>Drillstring Dynamics Simulator</strong> is your predictive planning tool. It allows you to test drilling parameters <em>before</em> applying them, preventing problems before they start. This represents the shift from reactive to proactive operations.
        <br><br>
        <strong>How a Drilling Professional Would Use This Tool:</strong>
        <ol>
            <li><strong>Planning:</strong> Before drilling a new, hard formation, an engineer inputs the expected conditions and planned parameters (e.g., Formation: 'Hard', RPM: 150, WOB: 60 kN) into the sidebar.</li>
            <li><strong>Analysis:</strong> They navigate to the 'Drillstring Dynamics Simulator' tab and observe a very high 'Stick-Slip Index' and an erratic torque wave on the graph. This indicates the planned parameters are risky and likely to cause damaging torsional instability.</li>
            <li><strong>Optimization:</strong> The engineer then interactively adjusts the parameters in the sidebar. They try reducing the RPM to 120 and increasing WOB to 70 kN. The simulation instantly reruns.</li>
            <li><strong>Result:</strong> The 'Stick-Slip Index' drops to a much lower value, and the torque wave on the graph becomes a smooth, stable sine wave. The engineer has now identified a safer, more stable set of parameters to provide to the rig crew, optimizing the operation and mitigating risk.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 3: Live Dashboard ---
with tab3:
    st.subheader("Real-Time Sensor Data & Alerts")
    placeholder = st.empty()

# --- Tab 4: Performance Overview ---
with tab4:
    st.subheader("Session Performance Summary")
    overview_placeholder = st.empty()

# --- Tab 5: Drillstring Dynamics Simulator ---
with tab5:
    st.header("🛠️ Drillstring Dynamics Dashboard")
    st.markdown("This tool simulates axial-torsional drillstring behavior. Adjust the parameters in the sidebar to see how they affect drillstring stability.")
    def run_drillstring_model(params):
        time_sim = np.linspace(0, 10, 1000); freq = params["rpm"] / 60
        friction_factor = {"Soft": 0.8, "Medium": 1.0, "Hard": 1.2}[params["formation"]]
        bit_modifier = {"PDC": 1.0, "Tricone": 0.9, "Diamond": 1.1}[params["bit_type"]]
        mud_modifier = {"Water-based": 1.0, "Oil-based": 1.05, "Synthetic": 1.1}[params["mud_type"]]
        stiffness_modifier = params["bha_stiffness"] / 50
        torque = params["wob"] * np.sin(2 * np.pi * freq * time_sim) * friction_factor * bit_modifier * mud_modifier
        displacement = np.cumsum(torque) * 0.01 / stiffness_modifier
        rpm_series = np.full_like(time_sim, params["rpm"]) + np.random.normal(0, 2, size=time_sim.shape)
        return time_sim, torque, displacement, rpm_series
    def compute_metrics(torque, rpm_series):
        ss_index = np.std(torque) / max(np.mean(torque), 1e-6)
        vibration_severity = np.max(np.abs(np.diff(rpm_series)))
        return round(ss_index, 3), round(vibration_severity, 3)
    sim_params = {"rpm": sim_rpm, "wob": sim_wob, "formation": sim_formation, "bit_type": sim_bit_type, "mud_type": sim_mud_type, "bha_stiffness": sim_bha_stiffness}
    time_sim, torque, displacement, rpm_series = run_drillstring_model(sim_params)
    ss_index, vibration_severity = compute_metrics(torque, rpm_series)
    st.subheader("Simulation Results")
    col1, col2 = st.columns(2)
    col1.metric("Stick-Slip Index", ss_index, help="A measure of torsional instability. Higher values indicate more severe stick-slip.")
    col2.metric("Vibration Severity", vibration_severity, help="The maximum rate of change in RPM, indicating axial vibration.")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(time_sim, torque, label="Torque (Nm)", color="dodgerblue"); ax.plot(time_sim, displacement, label="Axial Displacement (m)", color="darkorange")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Response"); ax.set_title("Drillstring Response Over Time"); ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig)
    results_df = pd.DataFrame({"Time (s)": time_sim, "Torque (Nm)": torque, "Displacement (m)": displacement, "RPM": rpm_series})
    csv = results_df.to_csv(index=False); st.download_button(label="📤 Download Results as CSV", data=csv, file_name="drillstring_results.csv", mime="text/csv")

# --- REAL-TIME LOOP ---
while True:
    new_timestamp = datetime.datetime.now()
    new_rpm = np.random.normal(loc=rpm_mean, scale=10)
    new_torque = np.random.normal(loc=500, scale=50)
    if np.random.rand() > 0.95: new_vibration = np.random.normal(loc=0.9, scale=0.2)
    else: new_vibration = np.random.normal(loc=0.5, scale=0.1)
    last_wear = st.session_state.drilling_data["Bit Wear Index"].iloc[-1] if not st.session_state.drilling_data.empty else 0
    new_bit_wear = np.clip(last_wear + np.random.normal(0.0005, 0.0002), 0, 1)
    new_rop = np.clip(100 - new_bit_wear * 80 + np.random.normal(0, 2), 20, 100)
    new_row = pd.DataFrame([{"Timestamp": new_timestamp, "RPM": new_rpm, "Torque": new_torque, "Vibration": new_vibration, "Bit Wear Index": new_bit_wear, "ROP (ft/hr)": new_rop}])
    st.session_state.drilling_data = pd.concat([st.session_state.drilling_data, new_row], ignore_index=True).tail(200)
    st.session_state.session_history = pd.concat([st.session_state.session_history, new_row], ignore_index=True)
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3); kpi1.metric(label="Live RPM", value=f"{new_rpm:.1f}"); kpi2.metric(label="Live Torque (Nm)", value=f"{new_torque:.1f}"); kpi3.metric(label="Live ROP (ft/hr)", value=f"{new_rop:.1f}")
        st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["RPM", "Torque"]], use_container_width=True); st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Vibration"]], use_container_width=True)
        st.subheader("⚠️ System Alerts")
        if new_vibration > vibration_threshold: st.warning(f"HIGH VIBRATION DETECTED! Current Level: {new_vibration:.2f}")
        else: st.success(f"Vibration levels normal. Current Level: {new_vibration:.2f}")
        st.subheader("📈 Optimization Insights"); st.line_chart(st.session_state.drilling_data.set_index("Timestamp")[["Bit Wear Index", "ROP (ft/hr)"]], use_container_width=True)
    with overview_placeholder.container():
        history = st.session_state.session_history; st.markdown("### Key Performance Indicators (Session Average)")
        avg_rop = history['ROP (ft/hr)'].mean(); avg_torque = history['Torque'].mean(); max_vibration = history['Vibration'].max()
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3); kpi_col1.metric("Average ROP (ft/hr)", f"{avg_rop:.2f}"); kpi_col2.metric("Average Torque (Nm)", f"{avg_torque:.2f}"); kpi_col3.metric("Max Vibration Recorded", f"{max_vibration:.2f}")
        st.markdown("<hr>", unsafe_allow_html=True); st.markdown("### Operational Status")
        col1, col2 = st.columns([1,2])
        with col1:
            high_vibration_count = history[history['Vibration'] > vibration_threshold].shape[0]; normal_count = history.shape[0] - high_vibration_count
            status_data = pd.DataFrame({'Status': ['Normal', 'High Vibration Alert'], 'Count': [normal_count, high_vibration_count]})
            fig_pie = px.pie(status_data, values='Count', names='Status', title='Time in Operational Status', color='Status', color_discrete_map={'Normal':'#2E8B57', 'High Vibration Alert':'#FF8C00'})
            fig_pie.update_layout(legend_title_text='Status', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig_pie, use_container_width=True)
        with col2: st.markdown("#### Dashboard Interpretation"); st.markdown("This overview provides a comprehensive analysis of the drilling session, focusing on efficiency and predictive maintenance.")
    time.sleep(1)
