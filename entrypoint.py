"""
Docker entrypoint: train the model if no artifact exists, then start the API.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL_PATH = Path("models") / "iot_anomaly_iforest.pkl"

if not MODEL_PATH.exists():
    print("[entrypoint] Model artifact not found -- running training pipeline ...")
    result = subprocess.run([sys.executable, "-m", "app.train_eval"], check=False)
    if result.returncode != 0:
        print("[entrypoint] ERROR: Training failed. Exiting.")
        sys.exit(1)
    print("[entrypoint] Training complete.")
else:
    print(f"[entrypoint] Model artifact found: {MODEL_PATH}")

print("[entrypoint] Starting Flask API on 0.0.0.0:5000 ...")
subprocess.run([sys.executable, "-m", "app.api"])
