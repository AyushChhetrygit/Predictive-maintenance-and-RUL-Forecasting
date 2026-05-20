import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Real-Time Monitoring", layout="wide")
st.title("📈 Real-Time Sensor Monitoring")

@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('data/features.pkl')
        return df
    except:
        st.error("Data not found. Please run the feature engineering pipeline first.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    engines = df['engine_id'].unique()
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_engine = st.selectbox("Select Engine", engines)
    with col2:
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
        
    engine_data = df[df['engine_id'] == selected_engine].sort_values('cycle')
    
    st.subheader(f"Trend for {selected_engine} - {selected_sensor}")
    
    # Calculate rolling statistics for anomaly highlighting
    window = 10
    rolling_mean = engine_data[selected_sensor].rolling(window=window).mean()
    rolling_std = engine_data[selected_sensor].rolling(window=window).std()
    
    # Define anomalies as points > 2 standard deviations from the rolling mean
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)
    
    anomalies = engine_data[(engine_data[selected_sensor] > upper_bound) | (engine_data[selected_sensor] < lower_bound)]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'], 
        y=engine_data[selected_sensor],
        mode='lines',
        name='Sensor Value',
        line=dict(color='blue')
    ))
    
    # Add anomalies as scatter points
    fig.add_trace(go.Scatter(
        x=anomalies['cycle'],
        y=anomalies[selected_sensor],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    # Highlight anomaly zones using shapes
    shapes = []
    for idx, row in anomalies.iterrows():
        shapes.append(
            dict(
                type="rect",
                xref="x", yref="paper",
                x0=row['cycle'] - 0.5, y0=0,
                x1=row['cycle'] + 0.5, y1=1,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )
    fig.update_layout(shapes=shapes)
    
    # Add annotations
    if not anomalies.empty:
        last_anomaly = anomalies.iloc[-1]
        fig.add_annotation(
            x=last_anomaly['cycle'],
            y=last_anomaly[selected_sensor],
            text="Latest Anomaly",
            showarrow=True,
            arrowhead=1
        )
        
    fig.update_layout(
        title=f"Real-Time Telemetry: {selected_sensor}",
        xaxis_title="Operational Cycle",
        yaxis_title="Sensor Value",
        hovermode="x unified",
        template="plotly_dark" # Modern dark UI
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Raw Anomaly Data"):
        st.dataframe(anomalies[['cycle', selected_sensor]])
