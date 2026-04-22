"""
Build a time-series plot from the live prediction log.
Writes app/outputs/anomalies_over_time.png.

Run:
    python -m app.visualize_from_csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LOG_PATH   = OUTPUT_DIR / "predictions_log.csv"

PALETTE = {"blue": "#2c5f8a", "red": "#e05555", "orange": "#e07b39"}


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {LOG_PATH}\n"
            "Run  python -m app.sender  to generate streaming data first."
        )
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Prediction log is empty.")
    return df


def save_anomalies_over_time(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("Streaming Predictions Over Time\nModel-to-Production IoT Anomalies",
                 fontsize=13, fontweight="bold")

    # top plot: score per prediction, split by normal vs anomaly
    normal    = df[df["is_anomaly"] == 0]
    anomalous = df[df["is_anomaly"] == 1]
    ax = axes[0]
    ax.scatter(normal["timestamp"], normal["anomaly_score"],
               s=14, alpha=0.5, color=PALETTE["blue"], label="Normal", zorder=2)
    ax.scatter(anomalous["timestamp"], anomalous["anomaly_score"],
               s=35, alpha=0.85, color=PALETTE["red"], label="Anomaly", zorder=5,
               marker="X")

    # threshold line
    if "threshold" in df.columns and df["threshold"].notna().any():
        thr = df["threshold"].dropna().iloc[0]
        ax.axhline(y=thr, color="gray", linestyle="--", lw=1.5,
                   label=f"Decision threshold ({thr:.4f})")

    ax.set_ylabel("Anomaly Score (lower = more anomalous)", fontsize=10)
    ax.set_title("Anomaly Score per Streamed Prediction", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=20)

    # bottom plot: rolling anomaly rate
    window = max(10, len(df) // 15)
    df_idx = df.set_index("timestamp").sort_index()
    roll = df_idx["is_anomaly"].rolling(window=window, min_periods=1).mean()
    ax2 = axes[1]
    ax2.plot(roll.index, roll.values * 100,
             color=PALETTE["orange"], lw=2.5, label=f"Rolling rate (window={window})")
    ax2.axhline(y=10, color="gray", linestyle="--", lw=1.5, label="Expected rate (10%)")
    ax2.set_ylabel("Anomaly Rate (%)", fontsize=10)
    ax2.set_xlabel("Time (UTC)", fontsize=10)
    ax2.set_title(f"Rolling Anomaly Rate  (window = {window} predictions)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=20)

    fig.tight_layout()
    out = OUTPUT_DIR / "anomalies_over_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> None:
    print("Visualising prediction log ...")
    df = load_log()
    n_total = len(df)
    n_anom  = int(df["is_anomaly"].sum())
    print(f"  Loaded {n_total:,} predictions  |  anomalies: {n_anom} ({n_anom/n_total:.1%})")
    save_anomalies_over_time(df)
    print("Done.")


if __name__ == "__main__":
    main()
