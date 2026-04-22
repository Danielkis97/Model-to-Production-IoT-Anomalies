# Model-to-Production IoT Anomalies

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black)
![Model](https://img.shields.io/badge/Model-IsolationForest-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Checks](https://img.shields.io/badge/Validation%20Checks-29%2F29-success)

This repository contains a prototype for stream-based IoT anomaly detection.
The scenario is a wind turbine component factory with three sensor channels:
`temperature`, `humidity`, `sound_volume`.

## System Architecture

![System Architecture Overview](docs/system_architecture_custom.png)

---

## Quick Repro

If you only want to verify the project quickly after cloning:

```bash
python -m app.train_eval
docker compose up --build
python tests/test.py
```

Expected result:
- model artifact + metadata are created
- API and sender run in Docker
- test script reports all checks passed

---

## Local Setup

```bash
git clone https://github.com/Danielkis97/Model-to-Production-IoT-Anomalies.git
cd Model-to-Production-IoT-Anomalies

python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                       # Linux / macOS

pip install -r requirements.txt
```

Run pipeline manually:

```bash
python -m app.train_eval
python -m app.api
python -m app.sender
python -m app.visualize_from_csv
python tests/test.py
```

---

## Docker Run

```bash
docker compose up --build
docker compose up -d
docker compose down
```

Services:

| Service | Purpose | Port |
|---|---|---|
| `iot-anomaly-api` | Flask REST API for prediction | `5000` |
| `iot-anomaly-sender` | Continuous sensor stream sender | outbound only |

---

## Project Structure

```text
Model-to-Production-IoT-Anomalies/
|-- app/
|   |-- api.py
|   |-- model.py
|   |-- sender.py
|   |-- train_eval.py
|   `-- visualize_from_csv.py
|-- app/outputs/
|-- data/
|-- models/
|-- metadata/
|-- tests/
|   `-- test.py
|-- docs/
|   |-- architecture.mmd
|   |-- system_architecture_custom.png
|   `-- screenshots/
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

---

## Workflow Summary

Training flow:

```text
simulate_sensor_data()
-> split 60/15/25
-> IsolationForest.fit(X_train)
-> threshold tuning on validation set
-> evaluate on held-out test set
-> save model + metadata + visuals
```

Inference flow:

```text
sender payload
-> POST /predict
-> score_samples + threshold decision
-> JSON response
-> append row to app/outputs/predictions_log.csv
```

---

## Metadata

The `metadata/` folder documents model, data, service, and project assumptions:

- `metadata/model_metadata.json`
- `metadata/dataset_metadata.json`
- `metadata/service_metadata.json`
- `metadata/project_metadata.md`

This makes the prototype easier to maintain and easier to review.

---

## API Endpoints

Main endpoints:

- `GET /health`
- `GET /model-info`
- `GET /metrics`
- `POST /predict`

Additional endpoint in implementation/tests:

- `POST /batch-predict`

---

## Execution Evidence

### Output: Training pipeline
![Training terminal output](docs/screenshots/training.png)

### Output: Docker build and launch
![Docker Compose build output](docs/screenshots/docker_build.png)

### Output: Live logs (API + sender)
![Docker Compose live logs](docs/screenshots/docker_logs.png)

### Output: System validation tests
![Test suite output](docs/screenshots/tests.png)

---

## Results and Visuals

Generated files (stored in `app/outputs/`):

- `metrics_table.png`
- `learning_dashboard_2x2.png`
- `heatmaps_grid.png`
- `histograms.png`
- `anomalies_over_time.png`

### Metrics Table
![Metrics table](app/outputs/metrics_table.png)

### Learning Dashboard (2x2)
![Learning dashboard](app/outputs/learning_dashboard_2x2.png)

### Correlation and Confusion Matrices
![Heatmaps grid](app/outputs/heatmaps_grid.png)

### Feature Histograms
![Feature histograms](app/outputs/histograms.png)

### Anomalies Over Time
![Anomalies over time](app/outputs/anomalies_over_time.png)

---

## Reproducibility Notes

- Split used by current implementation: **60% train / 15% validation / 25% test**.
- Threshold is tuned on validation F1, then applied unchanged to test and runtime predictions.
- Current test suite validates model files, API behavior, visuals, metadata, and Docker files.
- Latest validated test output: **PASS all 29 checks succeeded**.

---

## Development Environment

Validated locally with:

- Windows 11 (PowerShell)
- Python 3.11
- Docker Desktop + Docker Compose v2 (`docker compose`)
- Dependencies from `requirements.txt`

---

## Limitations

- Data is synthetic (fictional), not real factory sensor data.
- Flask development server is used in this prototype.
- Logging is CSV-based (`app/outputs/predictions_log.csv`), no production database.
- No authentication/authorization layer.
- No drift monitoring or scheduled retraining pipeline yet.
- No cloud deployment/hardening in this version.

---

## Conclusion

This project demonstrates an end-to-end model-to-production workflow for IoT anomaly detection:
training, artifact export, REST serving, stream ingestion, Docker reproducibility, and automated validation.
