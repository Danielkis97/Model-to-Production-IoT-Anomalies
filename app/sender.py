"""
IoT sensor stream simulator.

Pretends to be an edge device that reads sensor values once every few
seconds and posts them to the prediction API. Payload mix matches the
training anomaly rate (~10%).

Run:
    python -m app.sender
    python -m app.sender --count 60 --interval 1
    python -m app.sender --url http://localhost:5000 --count 100 --interval 0.5
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone

import requests

DEFAULT_URL      = "http://localhost:5000"
DEFAULT_INTERVAL = 2.0
DEFAULT_COUNT    = 0   # 0 = run forever


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IoT sensor stream sender")
    p.add_argument("--url",      default=DEFAULT_URL,
                   help="API base URL (default: http://localhost:5000)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help="Seconds between requests (default: 2)")
    p.add_argument("--count",    type=int, default=DEFAULT_COUNT,
                   help="Number of requests to send, 0 = forever (default: 0)")
    # kept so older docker-compose entries with --host/--port still work
    p.add_argument("--host", default=None, help=argparse.SUPPRESS)
    p.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _resolve_url(args: argparse.Namespace) -> str:
    if args.host is not None or args.port is not None:
        host = args.host or "localhost"
        port = args.port or 5000
        return f"http://{host}:{port}"
    return args.url


def _wait_for_api(base_url: str, timeout: int = 180) -> None:
    # poll /health until model is loaded
    print(f"[sender] Waiting for API at {base_url}/health ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("model_loaded"):
                print("[sender] API is ready.")
                return
            if r.status_code == 200:
                print("[sender] API up -- waiting for model to load ...")
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)
    print("[sender] ERROR: API did not become ready within timeout. Exiting.")
    sys.exit(1)


def _generate_payload(rng: random.Random) -> dict:
    # ~91% normal, ~9% anomalies (split across 3 types)
    roll = rng.random()

    if roll < 0.03:
        # overheating
        temp = rng.uniform(83.0, 99.0)
        hum  = rng.gauss(50.0, 6.5)
        snd  = rng.gauss(65.0, 4.5)
    elif roll < 0.06:
        # humidity surge
        temp = rng.gauss(70.0, 5.5)
        hum  = rng.uniform(71.0, 87.0)
        snd  = rng.gauss(65.0, 4.5)
    elif roll < 0.09:
        # sound spike / combined
        temp = rng.gauss(79.0, 5.0)
        hum  = rng.gauss(63.0, 5.0)
        snd  = rng.uniform(75.0, 92.0)
    elif roll < 0.16:
        # warm-up (normal but briefly elevated temperature)
        temp = rng.gauss(77.0, 4.0)
        hum  = rng.gauss(53.0, 5.0)
        snd  = rng.gauss(67.0, 3.5)
    else:
        # standard normal production
        temp = rng.gauss(70.0, 5.5)
        hum  = rng.gauss(50.0, 6.5)
        snd  = rng.gauss(65.0, 4.5)

    # add a bit of sensor noise
    temp += rng.gauss(0, 1.5)
    hum  += rng.gauss(0, 1.5)
    snd  += rng.gauss(0, 1.0)

    return {
        "temperature":  round(max(20.0, min(temp, 130.0)), 2),
        "humidity":     round(max(0.0,  min(hum,  100.0)), 2),
        "sound_volume": round(max(30.0, min(snd,  130.0)), 2),
    }


def run(args: argparse.Namespace) -> None:
    base_url = _resolve_url(args)
    _wait_for_api(base_url)

    rng       = random.Random()
    sent      = 0
    anomalies = 0
    count_lbl = "inf" if args.count == 0 else str(args.count)

    print(f"[sender] Streaming  url={base_url}  interval={args.interval}s  count={count_lbl}")
    sep = "-" * 82
    print(sep)
    print(f"{'Timestamp':22s}  {'Temp(C)':>8s}  {'Hum(%)':>7s}  {'Snd(dB)':>8s}"
          f"  {'Score':>10s}  {'Status':>13s}")
    print(sep)

    try:
        while args.count == 0 or sent < args.count:
            payload = _generate_payload(rng)
            try:
                resp = requests.post(f"{base_url}/predict", json=payload, timeout=5)
                ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

                if resp.status_code == 200:
                    data    = resp.json()
                    is_anom = data.get("is_anomaly", False)
                    score   = data.get("anomaly_score", 0.0)
                    tag     = "** ANOMALY **" if is_anom else "normal"
                    if is_anom:
                        anomalies += 1
                    print(
                        f"{ts:22s}  "
                        f"{payload['temperature']:8.2f}  "
                        f"{payload['humidity']:7.2f}  "
                        f"{payload['sound_volume']:8.2f}  "
                        f"{score:10.6f}  "
                        f"{tag:>13s}"
                    )
                else:
                    print(f"[sender] HTTP {resp.status_code}: {resp.text[:100]}")

            except requests.exceptions.RequestException as exc:
                print(f"[sender] Request error: {exc}")

            sent += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass

    print(sep)
    rate = anomalies / max(1, sent)
    print(f"[sender] Done  sent={sent}  anomalies={anomalies}  rate={rate:.1%}")


if __name__ == "__main__":
    run(_parse_args())
