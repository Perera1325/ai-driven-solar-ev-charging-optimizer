import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Solar EV Charging Optimizer", layout="wide")

st.title("⚡ AI Solar + EV Charging Optimizer Dashboard")

st.markdown("Live predictions powered by your ML + RL system")

# ---------------- Solar Prediction ----------------
st.header("☀️ Solar Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    ambient = st.number_input("Ambient Temperature (°C)", value=25.0)

with col2:
    module = st.number_input("Module Temperature (°C)", value=30.0)

with col3:
    irradiation = st.number_input("Irradiation", value=500.0)

if st.button("Predict Solar"):
    res = requests.post(
        f"{API_URL}/predict/solar",
        json={
            "ambient_temp": ambient,
            "module_temp": module,
            "irradiation": irradiation
        }
    )

    if res.status_code == 200:
        solar = res.json()["solar_power"]
        st.success(f"Predicted Solar Power: {solar:.2f}")
    else:
        st.error("Solar API not responding")

# ---------------- EV Prediction ----------------
st.header("🚗 EV Demand Prediction")

hour = st.slider("Hour", 0, 23, datetime.now().hour)
day = st.slider("Day (0=Mon)", 0, 6, datetime.now().weekday())

if st.button("Predict EV Demand"):
    res = requests.post(
        f"{API_URL}/predict/ev",
        json={"hour": hour, "day": day}
    )

    if res.status_code == 200:
        ev = res.json()["ev_demand"]
        st.success(f"Predicted EV Demand: {ev:.2f}")
    else:
        st.error("EV API not responding")

# ---------------- Simulation ----------------
st.header("🧠 Digital Twin Simulation (RL)")

if st.button("Run Simulation Step"):
    res = requests.get(f"{API_URL}/simulate")

    if res.status_code == 200:
        data = res.json()

        st.metric("Solar", round(data["solar"], 2))
        st.metric("EV Demand", round(data["ev_demand"], 2))
        st.metric("Battery Level", round(data["battery"], 2))
        st.metric("Action", data["action"])

    else:
        st.error("Simulation API not responding")

st.markdown("---")
st.caption("Built by Vinod — AI Driven Solar EV Charging Optimizer")
