⚡ AI-Driven Solar EV Charging Optimizer



An end-to-end AI system that optimizes electric vehicle charging using solar generation forecasts, EV demand prediction, a digital twin simulator, and reinforcement learning.



This project demonstrates a full ML + RL + Backend + Dashboard + Docker pipeline.



🚀 Features



☀️ Solar power forecasting (Machine Learning)



🚗 EV charging demand prediction (Machine Learning)



🔋 Digital twin of charging station



🧠 Reinforcement learning agent for charging optimization



🌐 FastAPI backend (REST API)



📊 Streamlit dashboard



🐳 Dockerized deployment



📈 Simulation logging + performance visualization



🏗 Architecture

Solar Dataset ──► Solar ML Model ─┐
                                ├──► Digital Twin ─► RL Agent ─► Decisions
EV Dataset ─────► EV ML Model ───┘

FastAPI Backend exposes predictions & simulation
Streamlit Dashboard visualizes results
Docker Compose runs everything




FastAPI Backend exposes predictions \& simulation

Streamlit Dashboard visualizes results

Docker Compose runs everything



🗂 Project Structure

ai-driven-solar-ev-charging-optimizer/

│

├── api/                # FastAPI backend

├── dashboard/         # Streamlit UI

├── data/              # Solar + EV datasets

├── models/            # Trained ML models

├── rl\_agent/          # Reinforcement learning agent

├── simulator/         # Digital twin + simulation

├── Dockerfile.api

├── Dockerfile.dashboard

├── docker-compose.yml

└── README.md



▶ Run Locally (Without Docker)

source venv/Scripts/activate

uvicorn api.main:app --reload





Second terminal:



streamlit run dashboard/app.py





API: http://127.0.0.1:8000/docs



Dashboard: http://localhost:8501



🐳 Run With Docker

docker compose build

docker compose up





Then open:



API: http://localhost:8000/docs



Dashboard: http://localhost:8501



📊 Simulation \& Graphs



Generate simulation:



python simulator/charging\_station.py





Plot results:



python simulator/plot\_results.py



🧠 Tech Stack



Python



Scikit-learn



Pandas / NumPy



FastAPI



Streamlit



Reinforcement Learning (Q-Learning)



Docker / Docker Compose



Matplotlib



👨‍💻 Author



Vinod Perera

Dual Degree Undergraduate — Computer Science \& Electrical Engineering

