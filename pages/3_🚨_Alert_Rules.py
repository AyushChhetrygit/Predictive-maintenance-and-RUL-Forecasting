import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="Alert Rules", layout="wide")
st.title("🚨 Alert Rule Builder")

st.markdown("Configure operational thresholds to trigger automated downtime risk notifications.")

RULES_FILE = 'output/alert_rules.json'

def load_rules():
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_rule(rule):
    rules = load_rules()
    rules.append(rule)
    os.makedirs('output', exist_ok=True)
    with open(RULES_FILE, 'w') as f:
        json.dump(rules, f, indent=4)

with st.form("alert_rule_builder"):
    st.subheader("Create a New Rule")
    
    col1, col2 = st.columns(2)
    with col1:
        sensor = st.selectbox("Select Target Metric/Sensor", [
            "Predicted RUL (Days)", 
            "sensor_11", "sensor_14", "sensor_15", "sensor_9"
        ])
        condition = st.selectbox("Condition", ["Less Than (<)", "Greater Than (>)", "Equals (==)"])
        threshold = st.number_input("Threshold Value", value=5.0)
        
    with col2:
        severity = st.select_slider("Severity Level", options=["Low", "Medium", "High", "Critical"])
        notification_channel = st.multiselect("Notification Channel", ["Email", "SMS", "Slack", "PagerDuty"])
        emails = st.text_input("Notify to (Comma separated emails)", value="admin@logicveda.com")
        
    submitted = st.form_submit_button("Create Rule")
    
    if submitted:
        new_rule = {
            "metric": sensor,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "channels": notification_channel,
            "contacts": emails
        }
        save_rule(new_rule)
        st.success(f"Rule successfully created for {sensor}!")

st.divider()

st.subheader("Active Alert Rules")
current_rules = load_rules()

if current_rules:
    df_rules = pd.DataFrame(current_rules)
    
    # Simple styler for the severity column
    def color_severity(val):
        color = 'green'
        if val == 'Medium': color = 'orange'
        elif val == 'High': color = 'orangered'
        elif val == 'Critical': color = 'red'
        return f'color: {color}; font-weight: bold'
        
    st.dataframe(df_rules.style.applymap(color_severity, subset=['severity']), use_container_width=True)
else:
    st.info("No active rules found. Create one above.")
