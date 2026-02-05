import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

print("Loading EV charging CSV data...")

# IMPORTANT: semicolon separated file
ev = pd.read_csv("data/Dataset 1_EV charging reports.csv", sep=";")

# Clean column names
ev.columns = ev.columns.str.strip()

print("Columns:")
print(ev.columns)

# Correct columns from YOUR dataset
time_col = "Start_plugin"
energy_col = "El_kWh"

# Keep only required columns
ev = ev[[time_col, energy_col]].dropna()

# Convert datetime
ev["start_time"] = pd.to_datetime(ev[time_col], errors="coerce")
ev = ev.dropna()

# Convert energy from "15,4" → 15.4
ev[energy_col] = ev[energy_col].astype(str).str.replace(",", ".", regex=False)
ev[energy_col] = ev[energy_col].astype(float)

# Feature engineering
ev["hour"] = ev["start_time"].dt.hour
ev["day"] = ev["start_time"].dt.dayofweek

X = ev[["hour", "day"]]
y = ev[energy_col]

print("Rows used:", len(ev))

print("Splitting EV data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training EV demand model...")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("EV MAE:", mae)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ev_model.pkl")

print("\n==============================")
print("EV MODEL TRAINED")
print("Saved to models/ev_model.pkl")
print("==============================")
