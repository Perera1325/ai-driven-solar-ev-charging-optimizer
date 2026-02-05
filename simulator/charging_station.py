import sys
sys.path.append(".")

import joblib
from datetime import datetime
from rl_agent.agent import ChargingAgent

# Load trained models
solar_model = joblib.load("models/solar_model.pkl")
ev_model = joblib.load("models/ev_model.pkl")

agent = ChargingAgent()

battery_capacity = 50.0
battery_level = 20.0

print("Digital Twin + RL Started")

def step(hour, day, temp=25, module_temp=30, irradiation=500):
    global battery_level

    # Predict solar + EV
    solar = solar_model.predict([[temp, module_temp, irradiation]])[0]
    ev = ev_model.predict([[hour, day]])[0]

    state = agent.get_state(battery_level, hour, day)
    action = agent.choose_action(state)

    # Action logic
    if action == 0:
        # Store solar
        battery_level += solar / 1000
        reward = solar * 0.01
    else:
        # Charge EV
        battery_level -= ev / 10
        reward = ev * 0.05

    battery_level = max(0, min(battery_capacity, battery_level))

    next_state = agent.get_state(battery_level, hour, day)
    agent.update(state, action, reward, next_state)

    return solar, ev, battery_level, action

# Run one simulation step
now = datetime.now()
hour = now.hour
day = now.weekday()

solar, ev, battery, action = step(hour, day)

print("Solar Generated:", round(solar, 2))
print("EV Demand:", round(ev, 2))
print("Battery Level:", round(battery, 2))
print("Action:", "Store Energy" if action == 0 else "Charge EV")
