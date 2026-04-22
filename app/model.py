"""
IsolationForest anomaly detector used by the Flask API.

Loads the trained model + tuned threshold from disk and exposes a
predict() method for single sensor readings.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT          = Path(__file__).resolve().parent.parent
MODEL_PATH    = ROOT / "models"   / "iot_anomaly_iforest.pkl"
METADATA_PATH = ROOT / "metadata" / "model_metadata.json"

FEATURES = ["temperature", "humidity", "sound_volume"]


class AnomalyDetector:
    # Wrapper around the trained IsolationForest. The decision threshold
    # is tuned on the validation set (F1-max sweep) and stored in metadata.

    def __init__(self) -> None:
        self._model = None
        self.metadata: dict = {}
        self.loaded: bool = False
        self._threshold: float | None = None

    def load(self) -> "AnomalyDetector":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}\n"
                "Run:  python -m app.train_eval"
            )
        self._model = joblib.load(MODEL_PATH)

        if METADATA_PATH.exists():
            with METADATA_PATH.open(encoding="utf-8") as fh:
                self.metadata = json.load(fh)

        self._threshold = self.metadata.get("threshold")
        self.loaded = True
        print(
            f"[model] Loaded {MODEL_PATH.name}  "
            f"version={self.metadata.get('model_version', '?')}  "
            f"threshold={self._threshold}"
        )
        return self

    def predict(self, temperature: float, humidity: float,
                sound_volume: float) -> dict:
        # score_samples: more negative = more anomalous.
        # If score < threshold we mark the reading as an anomaly.
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = np.array([[temperature, humidity, sound_volume]], dtype=float)
        score = float(self._model.score_samples(X)[0])

        if self._threshold is not None:
            is_anomaly = bool(score < self._threshold)
        else:
            # fallback to sklearn default if metadata is missing
            is_anomaly = bool(self._model.predict(X)[0] == -1)

        return {
            "is_anomaly":    is_anomaly,
            "anomaly_score": round(score, 6),
            "threshold":     round(self._threshold, 6) if self._threshold is not None else None,
        }


# single instance imported by app/api.py
detector = AnomalyDetector()
