import sys
sys.path.append(".")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from datetime import datetime
from rl_agent.agent import ChargingAgent

app = FastAPI(title="AI Solar EV Charging Optimizer")

solar_model = joblib.load("models/solar_model.pkl")
ev_model = joblib.load("models/ev_model.pkl")

agent = ChargingAgent()
battery_capacity = 50.0
battery_level = 20.0

class SolarInput(BaseModel):
    ambient_temp: float
    module_temp: float
    irradiation: float

class EVInput(BaseModel):
    hour: int
    day: int

@app.post("/predict/solar")
def predict_solar(data: SolarInput):
    pred = solar_model.predict([[data.ambient_temp, data.module_temp, data.irradiation]])[0]
    return {"solar_power": float(pred)}

@app.post("/predict/ev")
def predict_ev(data: EVInput):
    pred = ev_model.predict([[data.hour, data.day]])[0]
    return {"ev_demand": float(pred)}

@app.get("/simulate")
def simulate():
    global battery_level

    now = datetime.now()
    hour = now.hour
    day = now.weekday()

    solar = solar_model.predict([[25, 30, 500]])[0]
    ev = ev_model.predict([[hour, day]])[0]

    state = agent.get_state(battery_level, hour, day)
    action = agent.choose_action(state)

    if action == 0:
        battery_level += solar / 1000
    else:
        battery_level -= ev / 10

    battery_level = max(0, min(battery_capacity, battery_level))
    next_state = agent.get_state(battery_level, hour, day)
    agent.update(state, action, solar, next_state)

    return {
        "solar": float(solar),
        "ev_demand": float(ev),
        "battery": float(battery_level),
        "action": "store" if action == 0 else "charge"
    }
