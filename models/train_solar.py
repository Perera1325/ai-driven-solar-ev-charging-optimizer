import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

print("Loading CSV data...")

gen = pd.read_csv("data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

print("Converting DATE_TIME columns...")

gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"].str.strip())
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"].str.strip())

print("Merging datasets...")

data = pd.merge(gen, weather, on="DATE_TIME", how="inner")

print("Rows after merge:", len(data))

if len(data) == 0:
    raise Exception("Merged dataset is EMPTY. Check DATE_TIME columns.")

print("Cleaning data...")
data = data.dropna()

features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
target = "DC_POWER"

X = data[features]
y = data[target]

print("Splitting train / test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training RandomForest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating model...")

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("MAE:", mae)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/solar_model.pkl")

print("\n==============================")
print("TRAINING COMPLETE")
print("Model saved: models/solar_model.pkl")
print("==============================")
