"""
Flask REST API for the IoT anomaly detector.

Endpoints:
    GET  /health         liveness + model status
    GET  /model-info     model metadata (threshold, features, ...)
    GET  /metrics        runtime counters + average latency
    POST /predict        score one sensor reading
    POST /batch-predict  score a list of readings in one request

Run:
    python -m app.api
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

from app.model import FEATURES, detector

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "predictions_log.csv"
MODEL_METADATA_PATH = ROOT / "metadata" / "model_metadata.json"

_LOG_FIELDS = [
    "timestamp", "temperature", "humidity", "sound_volume",
    "anomaly_score", "threshold", "is_anomaly", "latency_ms",
]

# simple in-memory counters exposed via /metrics
_metrics: dict = {
    "request_count":    0,
    "prediction_count": 0,
    "anomaly_count":    0,
    "normal_count":     0,
    "total_latency_ms": 0.0,
}


# --- helpers ------------------------------------------------------------------

def _append_log(row: dict) -> None:
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_features(data: dict) -> tuple[float, float, float]:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return (
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["sound_volume"]),
    )


def _run_prediction(temperature: float, humidity: float,
                    sound_volume: float) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = detector.predict(temperature, humidity, sound_volume)
    latency_ms = (time.perf_counter() - t0) * 1000

    _metrics["prediction_count"] += 1
    _metrics["total_latency_ms"] += latency_ms
    if result["is_anomaly"]:
        _metrics["anomaly_count"] += 1
    else:
        _metrics["normal_count"] += 1

    return result, latency_ms


def _load_model_metadata_from_file() -> dict:
    if not MODEL_METADATA_PATH.exists():
        return {}
    try:
        with MODEL_METADATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# --- endpoints ----------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    # always 200 so docker healthchecks can poll readiness
    return jsonify({
        "status":       "ok",
        "model_loaded": detector.loaded,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "service_name": "Model-to-Production IoT Anomalies",
        "version":      "1.0.0",
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    if not detector.loaded:
        return jsonify({"error": "Model not loaded -- run training first"}), 503
    file_metadata = _load_model_metadata_from_file()
    return jsonify(file_metadata or detector.metadata)


@app.route("/metrics", methods=["GET"])
def metrics():
    pc = _metrics["prediction_count"]
    avg_lat = _metrics["total_latency_ms"] / pc if pc > 0 else 0.0
    return jsonify({
        "request_count":      _metrics["request_count"],
        "prediction_count":   pc,
        "anomaly_count":      _metrics["anomaly_count"],
        "normal_count":       _metrics["normal_count"],
        "average_latency_ms": round(avg_lat, 3),
        "model_loaded":       detector.loaded,
    })


@app.route("/predict", methods=["POST"])
def predict():
    _metrics["request_count"] += 1

    if not detector.loaded:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        temperature, humidity, sound_volume = _parse_features(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    result, latency_ms = _run_prediction(temperature, humidity, sound_volume)
    ts = datetime.now(timezone.utc).isoformat()

    _append_log({
        "timestamp":    ts,
        "temperature":  temperature,
        "humidity":     humidity,
        "sound_volume": sound_volume,
        "anomaly_score": result["anomaly_score"],
        "threshold":    result["threshold"],
        "is_anomaly":   int(result["is_anomaly"]),
        "latency_ms":   round(latency_ms, 3),
    })

    return jsonify({
        "is_anomaly":    result["is_anomaly"],
        "anomaly_score": result["anomaly_score"],
        "threshold":     result["threshold"],
        "model_version": detector.metadata.get("model_version", "1.0.0"),
        "timestamp":     ts,
        "input": {
            "temperature":  temperature,
            "humidity":     humidity,
            "sound_volume": sound_volume,
        },
    })


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    _metrics["request_count"] += 1

    if not detector.loaded:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(silent=True)
    if not isinstance(body, list):
        return jsonify({"error": "Expected a JSON array of sensor records"}), 400

    ts = datetime.now(timezone.utc).isoformat()
    results = []
    for item in body:
        try:
            t, h, s = _parse_features(item)
        except (ValueError, TypeError) as exc:
            results.append({"error": str(exc)})
            continue
        r, _ = _run_prediction(t, h, s)
        results.append({**r, "timestamp": ts})

    return jsonify(results)


# --- startup ------------------------------------------------------------------

def _load_model_at_startup() -> None:
    try:
        detector.load()
    except FileNotFoundError as exc:
        print(f"[api] WARNING -- {exc}")
        print("[api] /health will report model_loaded=false until training runs.")


_load_model_at_startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
