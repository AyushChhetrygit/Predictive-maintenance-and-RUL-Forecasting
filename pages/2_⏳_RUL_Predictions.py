import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

st.set_page_config(page_title="RUL Predictions", layout="wide")
st.title("⏳ Remaining Useful Life (RUL) Forecast")

# Load dummy schedule / predictions to simulate model output
@st.cache_data
def load_predictions():
    try:
        with open('output/optimal_schedule.json', 'r') as f:
            data = json.load(f)
        return data['schedule']
    except:
        return {"engine_1": 15, "engine_2": 8, "engine_3": 2}

rul_data = load_predictions()
engines = list(rul_data.keys())

selected_engine = st.selectbox("Select Engine", engines)

current_rul = rul_data.get(selected_engine, 10)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current RUL Gauge")
    
    # Determine color based on RUL severity
    if current_rul > 10:
        bar_color = "green"
    elif current_rul > 5:
        bar_color = "orange"
    else:
        bar_color = "red"
        
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_rul,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Days Remaining"},
        gauge = {
            'axis': {'range': [None, 30]},
            'bar': {'color': bar_color},
            'steps' : [
                {'range': [0, 5], 'color': "lightcoral"},
                {'range': [5, 10], 'color': "navajowhite"},
                {'range': [10, 30], 'color': "lightgreen"}
            ],
            'threshold' : {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 2
            }
        }
    ))
    fig_gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.subheader("RUL Degradation Forecast")
    
    # Simulate historical RUL degradation for visualization
    cycles = list(range(1, 21))
    # Simulated linear degradation + some noise
    rul_trend = [current_rul + (20 - c) * 0.8 + (np.random.rand() * 2 - 1) for c in cycles]
    upper_band = [r + (r * 0.15) for r in rul_trend] # 15% confidence interval
    lower_band = [r - (r * 0.15) for r in rul_trend]
    
    fig_line = go.Figure()

    # Confidence Band
    fig_line.add_trace(go.Scatter(
        x=cycles + cycles[::-1], # x, then x reversed
        y=upper_band + lower_band[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))

    # Mean RUL Prediction Line
    fig_line.add_trace(go.Scatter(
        x=cycles,
        y=rul_trend,
        mode='lines+markers',
        name='Predicted RUL',
        line=dict(color='cyan', width=3)
    ))
    
    # Critical threshold
    fig_line.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Critical Action Required", annotation_position="bottom right")

    fig_line.update_layout(
        xaxis_title="Operational Cycle",
        yaxis_title="RUL (Days)",
        hovermode="x unified",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
