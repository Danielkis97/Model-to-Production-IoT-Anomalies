# Project Metadata

## Project
- Name: Model-to-Production IoT Anomalies
- Course: IU DLBDSMTP01 - Project: From Model to Production
- Task: Task 1 - Anomaly detection in an IoT setting (spotlight: stream processing)
- Scenario: Wind turbine component manufacturing

## Purpose
This project demonstrates how a simple anomaly detection model can be moved into a production-oriented service prototype.

## Model
- IsolationForest is used as the anomaly detector.
- It was chosen because it works without requiring class labels during training.
- The model uses `temperature`, `humidity`, and `sound_volume`.
- Anomaly detection fits this task because unusual sensor patterns indicate potential production faults.

## Data
- The dataset is synthetic and fictional.
- Features: `temperature`, `humidity`, `sound_volume`.
- The simulation includes noise, overlap, and borderline cases.
- Metrics are intentionally not perfect to reflect realistic uncertainty.

## Service
- Flask REST API serves predictions.
- The model is loaded at API startup.
- A sender process streams simulated sensor data to `/predict`.
- Predictions are logged and can be monitored through log files and generated metrics.

## Reproducibility
- `README.md` documents setup and run commands.
- Docker artifacts (`Dockerfile`, `docker-compose.yml`) provide reproducible execution.
- `tests/test.py` validates end-to-end behavior and required files.
- Generated outputs in `app/outputs` support result traceability.

## Limitations
- Synthetic data only.
- No real production deployment in this scope.
- No authentication or authorization.
- CSV-based logging only.
- No drift monitoring.
- No scheduled retraining.

## Presentation Use
These metadata files support the oral project report by making model versioning, data assumptions, service design, and limitations traceable.
