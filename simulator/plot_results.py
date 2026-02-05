import pandas as pd
import matplotlib.pyplot as plt

# Load simulation log
df = pd.read_csv("simulator/simulation_log.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# ---- Plot Solar, EV, Battery ----
plt.figure()
plt.plot(df["time"], df["solar"], label="Solar")
plt.plot(df["time"], df["ev_demand"], label="EV Demand")
plt.plot(df["time"], df["battery"], label="Battery")
plt.legend()
plt.title("Solar vs EV vs Battery (24 Steps)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Plot RL Actions ----
action_counts = df["action"].value_counts()

plt.figure()
action_counts.plot(kind="bar")
plt.title("RL Actions Distribution")
plt.show()
