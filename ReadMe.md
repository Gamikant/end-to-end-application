# 🚀 MLOps Fraud Detection Platform

A modular, reproducible machine learning operations (MLOps) platform for fraud detection.  
Supports experiment tracking, data versioning, and an interactive web UI for non-technical and technical users.

---

## Features

- 📊 **Experiment Tracking:** MLflow integration for metrics, parameters, and artifact logging  
- 🗃️ **Data Versioning:** DVC for datasets and model artifacts  
- 🖥️ **Web Dashboard:** React frontend for uploading data, launching runs, and viewing results  
- 🐳 **Containerized:** All components run in Docker for easy setup  
- 🏷️ **API-first:** REST endpoints for automation and integration

---

## Architecture

- **Frontend:** React (client)
- **Backend:** Node.js + Express (server)
- **ML Pipelines:** Python (invoked by backend)
- **Database:** MongoDB (for run metadata)
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Orchestration:** Docker Compose

---

## Quickstart

### 1. **Clone the repository**

```bash
git clone git@github.com:Gamikant/Feature-Selection.git
cd Feature-Selection
```

### 2. **Prepare your data**

- Place your training and test CSV files in `server/uploads` (or upload via the UI).

### 3. **Start the platform**

```bash
docker-compose up --build
```

- The first run may take a few minutes to download images and install dependencies.

### 4. **Access the application**

- **Web UI:** [http://localhost:3000](http://localhost:3000)
- **MLflow UI:** [http://localhost:5001](http://localhost:5001)
- **MongoDB:** [localhost:27017](mongodb://localhost:27017) (if you want to connect directly)

---

## Usage

### **Running a New Experiment**

1. Go to [http://localhost:3000](http://localhost:3000)
2. Click "Fraud Detection ML Platform"
3. Upload your train and test data (CSV)
4. Click "Train & Predict"
5. Feature Selection and Test Results will be shown in the respective tabs
6. Note Down the Run ID and you can search it later on in Previous Runs

### **Viewing Results**

- **Feature Selection:** See which features were used/dropped
- **Test Results:** View confusion matrix, fraud statistics, and download flagged transactions

### **Tracking Experiments**

- Open [http://localhost:5001](http://localhost:5001) for the MLflow UI
- Compare runs, download models, and review metrics

---

## Guides & Documentation

See [User Manual](docs/User-Manual.pdf) for full details for running the website.
See [Low Level Design Document](docs/Low-Level-Design.pdf) for full API details.
See [High Level Design Document](docs/High-Level-Design.pdf) for full structure.

---

## Developer Notes

- **Hot reload**: Changes to frontend or backend code will auto-reload in development mode.
- **Data & models**: All data, models, and artifacts are versioned with DVC and stored in `/dvc-storage` (mounted as a Docker volume).
- **MLflow artifacts**: Stored in `./mlruns` (mounted as a Docker volume).

---

## Troubleshooting

- **Port conflicts:** Ensure ports 3000, 5000, 5001, and 27017 are free.
- **First-time setup:** If you see missing dependencies, try `docker-compose build --no-cache`.
- **Clearing data:** To reset all data and containers:
  ```bash
  docker-compose down -v
  ```

---
