import sys
sys.path.append(".")

import joblib
from datetime import datetime, timedelta
import pandas as pd
from rl_agent.agent import ChargingAgent

solar_model = joblib.load("models/solar_model.pkl")
ev_model = joblib.load("models/ev_model.pkl")

agent = ChargingAgent()

battery_capacity = 50.0
battery_level = 20.0

print("Digital Twin + RL Started (Logging Enabled)")

def step(hour, day, temp=25, module_temp=30, irradiation=500):
    global battery_level

    solar = solar_model.predict([[temp, module_temp, irradiation]])[0]
    ev = ev_model.predict([[hour, day]])[0]

    state = agent.get_state(battery_level, hour, day)
    action = agent.choose_action(state)

    if action == 0:   # store
        battery_level += solar / 1000
        reward = solar * 0.01
    else:             # charge
        battery_level -= ev / 10
        reward = ev * 0.05

    battery_level = max(0, min(battery_capacity, battery_level))

    next_state = agent.get_state(battery_level, hour, day)
    agent.update(state, action, reward, next_state)

    return solar, ev, battery_level, action

# -------- Run multi-step simulation (24 steps) --------
rows = []
now = datetime.now()

for i in range(24):
    t = now + timedelta(hours=i)
    hour = t.hour
    day = t.weekday()

    solar, ev, battery, action = step(hour, day)

    rows.append({
        "time": t.isoformat(),
        "solar": float(solar),
        "ev_demand": float(ev),
        "battery": float(battery),
        "action": "store" if action == 0 else "charge"
    })

df = pd.DataFrame(rows)
df.to_csv("simulator/simulation_log.csv", index=False)

print("Saved simulator/simulation_log.csv")
