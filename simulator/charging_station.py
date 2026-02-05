import joblib
from datetime import datetime

# Load trained models
solar_model = joblib.load("models/solar_model.pkl")
ev_model = joblib.load("models/ev_model.pkl")

battery_capacity = 50.0  # kWh
battery_level = 20.0

print("Digital Twin Started")

def simulate_step(hour, day, temp=25, module_temp=30, irradiation=500):
    global battery_level

    # Solar prediction
    solar_input = solar_model.predict([[temp, module_temp, irradiation]])[0]

    # EV demand prediction
    ev_demand = ev_model.predict([[hour, day]])[0]

    # Update battery
    battery_level += solar_input / 1000
    battery_level -= ev_demand / 10

    battery_level = max(0, min(battery_capacity, battery_level))

    return solar_input, ev_demand, battery_level

# Run simulation
now = datetime.now()
hour = now.hour
day = now.weekday()

solar, ev, battery = simulate_step(hour, day)

print("Solar Generated:", round(solar, 2))
print("EV Demand:", round(ev, 2))
print("Battery Level:", round(battery, 2))
