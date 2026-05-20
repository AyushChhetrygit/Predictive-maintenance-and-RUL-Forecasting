import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown("""
Welcome to the LogicVeda Predictive Maintenance Dashboard!

This production dashboard monitors the health of our engine fleet, forecasts failure (RUL), and optimizes maintenance schedules.

### Features
👈 **Select a page from the sidebar to explore:**

*   **📈 Real-Time Monitoring:** Live sensor data with automatic anomaly detection and highlighting.
*   **⏳ RUL Predictions:** Machine Learning-driven Remaining Useful Life forecasts with confidence intervals.
*   **🚨 Alert Rules:** Configure customizable alerting rules for engine anomalies and downtime risks.
*   **📊 Data Drift:** AI system health monitoring using Evidently AI to detect feature and target drift.

***
*LogicVeda End-to-End Simulation Pipeline*
""")

# Setup some session state for shared data across pages
if 'engines' not in st.session_state:
    st.session_state['engines'] = ["engine_1", "engine_2", "engine_3", "engine_4", "engine_5"]

