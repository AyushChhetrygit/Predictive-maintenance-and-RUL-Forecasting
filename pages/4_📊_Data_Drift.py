import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import nest_asyncio
nest_asyncio.apply()

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
except ModuleNotFoundError:
    # Fallback for newer evidently versions
    from evidently import Report
    from evidently.presets import DataDriftPreset

st.set_page_config(page_title="Data Drift Monitoring", layout="wide")
st.title("📊 Model Health & Data Drift")

st.markdown("""
Continuous monitoring of incoming sensor data is critical. We use **Evidently AI** to perform 
Kolmogorov-Smirnov (KS-tests) on feature distributions to detect if the new data differs significantly from our training baseline.
""")

@st.cache_data
def generate_drift_report():
    try:
        df = pd.read_pickle('data/features.pkl')
        
        # Simulate Reference vs Current data by splitting the dataset
        # In production, Reference is training data, Current is the past week's incoming telemetry
        
        # We will split engines: half reference, half current to guarantee some distribution differences
        engines = df['engine_id'].unique()
        midpoint = len(engines) // 2
        
        reference_data = df[df['engine_id'].isin(engines[:midpoint])]
        current_data = df[df['engine_id'].isin(engines[midpoint:])]
        
        # To make it run fast in a Streamlit demo, we'll sample down to a few thousand rows
        if len(reference_data) > 5000:
            reference_data = reference_data.sample(5000)
        if len(current_data) > 5000:
            current_data = current_data.sample(5000)
            
        sensor_cols = [c for c in df.columns if 'sensor' in c]
        
        # Build the Evidently Report
        report = Report(metrics=[
            DataDriftPreset(),
            # TargetDriftPreset() # We can add target drift if RUL is present
        ])
        
        # Run report calculation
        # To avoid passing unnecessary string cols that cause errors, keep only numerical sensor cols
        snapshot = report.run(reference_data=reference_data[sensor_cols], 
                   current_data=current_data[sensor_cols])
        
        # Save HTML
        os.makedirs('output', exist_ok=True)
        report_path = 'output/evidently_drift_report.html'
        
        # Support both new and old Evidently APIs
        if snapshot and hasattr(snapshot, 'save_html'):
            snapshot.save_html(report_path)
        else:
            report.save_html(report_path)
        
        return report_path
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

with st.spinner("Generating Weekly Data Drift Report... (This may take a minute)"):
    report_file = generate_drift_report()

if report_file and os.path.exists(report_file):
    st.success("✅ Drift Report Generated Successfully")
    
    with open(report_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    # Render the Evidently HTML report directly in the Streamlit app
    components.html(html_content, height=1000, scrolling=True)
else:
    st.warning("Could not load the Evidently report.")
