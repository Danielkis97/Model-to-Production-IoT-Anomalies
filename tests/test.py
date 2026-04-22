"""
System validation tests.

Checks:
  - model files exist
  - all 5 API endpoints respond as expected
  - input validation returns 400
  - prediction log is written
  - all 5 PNG visualisations are present
  - Dockerfile, docker-compose.yml, README, architecture diagram exist

Prerequisites:
  1. python -m app.train_eval
  2. python -m app.api   (keep running in a separate terminal)
  3. python tests/test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT       = Path(__file__).resolve().parent.parent
API_URL    = "http://localhost:5000"
OUTPUT_DIR = ROOT / "app" / "outputs"
META_DIR   = ROOT / "metadata"

_results: list[tuple[str, bool]] = []


def _ok(label: str, detail: str = "") -> None:
    msg = f"[test] {label} OK"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((label, True))


def _fail(label: str, reason: str = "") -> None:
    msg = f"[FAIL] {label}"
    if reason:
        msg += f"  -- {reason}"
    print(msg)
    _results.append((label, False))


def _check(label: str, passed: bool, detail: str = "", reason: str = "") -> None:
    if passed:
        _ok(label, detail)
    else:
        _fail(label, reason)


def test_model_files() -> None:
    pkl  = ROOT / "models"   / "iot_anomaly_iforest.pkl"
    meta = ROOT / "metadata" / "model_metadata.json"
    _check("model files", pkl.exists() and meta.exists(),
           reason="run python -m app.train_eval first" if not pkl.exists() else "")


def test_health() -> None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        _check("health", r.status_code == 200 and data.get("status") == "ok",
               reason=f"HTTP {r.status_code}")
        _check("model loaded", bool(data.get("model_loaded")),
               reason="model_loaded is false")
    except Exception as exc:
        _fail("health", str(exc))
        _fail("model loaded", "API unreachable")


def test_model_info() -> None:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        data = r.json() if r.ok else {}
        required = ["model_type", "model_version", "features",
                    "contamination", "threshold", "training_date", "data_type"]
        ok = r.ok and all(k in data for k in required)
        _check("model-info", ok,
               detail=f"threshold={data.get('threshold', '?')}",
               reason="missing fields" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        _fail("model-info", str(exc))


def test_metrics() -> None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        data = r.json() if r.ok else {}
        required = ["request_count", "prediction_count", "anomaly_count",
                    "normal_count", "average_latency_ms", "model_loaded"]
        _check("metrics", r.ok and all(k in data for k in required),
               reason="missing fields" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        _fail("metrics", str(exc))


def test_predict_valid() -> None:
    payload = {"temperature": 70.0, "humidity": 50.0, "sound_volume": 65.0}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        data = r.json() if r.ok else {}
        required = ["is_anomaly", "anomaly_score", "threshold",
                    "model_version", "timestamp", "input"]
        ok = r.ok and all(k in data for k in required)
        _check("predict valid", ok,
               detail=f"is_anomaly={data.get('is_anomaly','?')}  score={data.get('anomaly_score','?')}",
               reason="missing fields" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        _fail("predict valid", str(exc))


def test_predict_anomaly() -> None:
    # extreme values -> should be flagged
    payload = {"temperature": 100.0, "humidity": 88.0, "sound_volume": 93.0}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        data = r.json() if r.ok else {}
        _check("predict anomaly", r.ok and data.get("is_anomaly", False),
               detail=f"score={data.get('anomaly_score','?')}",
               reason="not flagged as anomaly" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        _fail("predict anomaly", str(exc))


def test_predict_invalid() -> None:
    # missing fields -> 400
    r1 = requests.post(f"{API_URL}/predict",
                       json={"temperature": 70.0}, timeout=5)
    _check("predict missing fields -> 400", r1.status_code == 400,
           reason=f"got {r1.status_code}")

    # bad type -> 400
    r2 = requests.post(f"{API_URL}/predict",
                       json={"temperature": "hot", "humidity": 50, "sound_volume": 65},
                       timeout=5)
    _check("predict bad type -> 400", r2.status_code == 400,
           reason=f"got {r2.status_code}")


def test_batch_predict_valid() -> None:
    payload = [
        {"temperature": 70.0,  "humidity": 50.0, "sound_volume": 65.0},
        {"temperature": 100.0, "humidity": 88.0, "sound_volume": 93.0},
        {"temperature": 72.0,  "humidity": 48.0, "sound_volume": 64.0},
    ]
    try:
        r = requests.post(f"{API_URL}/batch-predict", json=payload, timeout=5)
        data = r.json() if r.ok else []
        ok = (r.ok and isinstance(data, list) and len(data) == 3
              and all("is_anomaly" in d for d in data))
        flags = [d.get("is_anomaly") for d in data] if ok else []
        _check("batch-predict valid", ok,
               detail=f"flags={flags}",
               reason="bad response structure" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        _fail("batch-predict valid", str(exc))


def test_batch_predict_invalid() -> None:
    # not an array -> 400
    r = requests.post(f"{API_URL}/batch-predict",
                      json={"temperature": 70, "humidity": 50, "sound_volume": 65},
                      timeout=5)
    _check("batch-predict non-array -> 400", r.status_code == 400,
           reason=f"got {r.status_code}")


def test_prediction_log() -> None:
    log = OUTPUT_DIR / "predictions_log.csv"
    _check("prediction log", log.exists(),
           detail=f"{log.stat().st_size} bytes" if log.exists() else "",
           reason="file missing -- run sender or make a /predict call first")


def test_visuals() -> None:
    for fname in [
        "metrics_table.png",
        "learning_dashboard_2x2.png",
        "heatmaps_grid.png",
        "histograms.png",
        "anomalies_over_time.png",
    ]:
        p  = OUTPUT_DIR / fname
        ok = p.exists() and p.stat().st_size > 5_000
        _check(f"visual {fname}", ok,
               detail=f"{p.stat().st_size // 1024} KB" if ok else "",
               reason="missing or too small -- run train_eval and visualize_from_csv")


def test_docker_files() -> None:
    for fname in ["Dockerfile", "docker-compose.yml"]:
        p = ROOT / fname
        _check(f"docker file {fname}", p.exists(), reason="file missing")


def test_readme() -> None:
    p = ROOT / "README.md"
    _check("README.md", p.exists() and p.stat().st_size > 500,
           reason="missing or too short")


def test_architecture() -> None:
    p = ROOT / "docs" / "architecture.mmd"
    _check("architecture.mmd", p.exists(), reason="missing")


def test_metadata_files() -> None:
    for fname in [
        "model_metadata.json",
        "dataset_metadata.json",
        "service_metadata.json",
        "project_metadata.md",
    ]:
        p = META_DIR / fname
        _check(f"metadata {fname}", p.exists(), reason="file missing")


def test_model_metadata_content() -> None:
    p = META_DIR / "model_metadata.json"
    if not p.exists():
        _fail("model_metadata content", "model_metadata.json missing")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail("model_metadata content", str(exc))
        return
    required = ["project_name", "model_type", "feature_names", "threshold"]
    _check("model_metadata content", all(k in data for k in required), reason="missing fields")


def test_dataset_metadata_content() -> None:
    p = META_DIR / "dataset_metadata.json"
    if not p.exists():
        _fail("dataset_metadata content", "dataset_metadata.json missing")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail("dataset_metadata content", str(exc))
        return
    ok = "data_source" in data and isinstance(data.get("features"), dict)
    _check("dataset_metadata content", ok, reason="missing data_source or features")


def test_service_metadata_content() -> None:
    p = META_DIR / "service_metadata.json"
    if not p.exists():
        _fail("service_metadata content", "service_metadata.json missing")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail("service_metadata content", str(exc))
        return
    endpoints = data.get("endpoints")
    _check("service_metadata endpoints", isinstance(endpoints, dict), reason="missing endpoints")


def test_project_metadata_content() -> None:
    p = META_DIR / "project_metadata.md"
    if not p.exists():
        _fail("project_metadata content", "project_metadata.md missing")
        return
    text = p.read_text(encoding="utf-8")
    _check("project_metadata title", "Model-to-Production IoT Anomalies" in text,
           reason="project name missing")


def main() -> None:
    print("=" * 60)
    print("  Model-to-Production IoT Anomalies")
    print("  System Validation Tests")
    print(f"  API: {API_URL}")
    print("=" * 60)
    print()

    test_model_files()
    print()
    test_health()
    print()
    test_model_info()
    test_metrics()
    print()
    test_predict_valid()
    test_predict_anomaly()
    test_predict_invalid()
    test_batch_predict_valid()
    test_batch_predict_invalid()
    print()
    test_prediction_log()
    print()
    test_visuals()
    print()
    test_docker_files()
    test_readme()
    test_architecture()
    print()
    test_metadata_files()
    test_model_metadata_content()
    test_dataset_metadata_content()
    test_service_metadata_content()
    test_project_metadata_content()

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    if passed == total:
        print(f"[test] PASS  all {total} checks succeeded")
    else:
        failed = total - passed
        print(f"[FAIL] {passed}/{total} passed -- {failed} check(s) failed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
