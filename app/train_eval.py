"""
Training + evaluation pipeline for the IoT anomaly detector.

Steps:
    1. Generate synthetic sensor data (5000 records)
    2. 60/15/25 split -> train / val / test
    3. Fit IsolationForest on the train split
    4. Tune the decision threshold on val (F1 sweep)
    5. Evaluate on the held-out test split
    6. Save the model + all visualisations

Run:
    python -m app.train_eval

Outputs:
    models/iot_anomaly_iforest.pkl
    metadata/model_metadata.json
    metadata/dataset_metadata.json
    data/train.csv, data/val.csv, data/test.csv
    app/outputs/metrics_table.csv
    app/outputs/metrics_table.png
    app/outputs/learning_dashboard_2x2.png
    app/outputs/heatmaps_grid.png
    app/outputs/histograms.png
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "app" / "outputs"
MODEL_DIR  = ROOT / "models"
META_DIR   = ROOT / "metadata"
DATA_DIR   = ROOT / "data"

for _d in [OUTPUT_DIR, MODEL_DIR, META_DIR, DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED    = 42
N_SAMPLES      = 5_000
ANOMALY_RATE   = 0.10     # 10% of dataset is anomalous
CONTAMINATION  = 0.10     # IsolationForest param, matches injection rate
N_ESTIMATORS   = 100
MODEL_VERSION  = "1.0.0"
FEATURES       = ["temperature", "humidity", "sound_volume"]
PROJECT_NAME   = "Model-to-Production IoT Anomalies"
SCENARIO       = "Wind turbine component manufacturing"
DATA_TYPE      = "synthetic / fictional sample data"
SPLIT          = {"train": "60%", "validation": "15%", "test": "25%"}


# --- 1. data simulation -------------------------------------------------------

def simulate_sensor_data(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a synthetic factory sensor dataset (3 channels, 5 anomaly types).

    Normal production is wide/noisy on purpose so the model can't trivially
    separate classes. Anomaly tails overlap with the upper end of the normal
    distribution, which keeps metrics realistic.
    """
    rng = np.random.default_rng(seed)

    n_anomaly = int(n * ANOMALY_RATE)
    n_normal  = n - n_anomaly

    # how many samples per anomaly type
    n1 = int(n_anomaly * 0.35)   # overheating
    n2 = int(n_anomaly * 0.25)   # humidity surge
    n3 = int(n_anomaly * 0.20)   # sound / vibration
    n4 = int(n_anomaly * 0.15)   # combined mild
    n5 = n_anomaly - n1 - n2 - n3 - n4   # borderline (rest)

    # normal readings: 96% base + 4% warm-up phase
    n_base   = int(n_normal * 0.96)
    n_warmup = n_normal - n_base

    t_base = rng.normal(70.0, 5.0,  n_base)
    h_base = rng.normal(50.0, 6.0,  n_base)
    s_base = rng.normal(65.0, 4.0,  n_base)

    # warm-up: elevated temp but still normal (NOT a fault)
    t_wu = rng.normal(76.0, 3.5, n_warmup)
    h_wu = rng.normal(52.0, 4.5, n_warmup)
    s_wu = rng.normal(67.0, 3.0, n_warmup)

    temp_n = np.concatenate([t_base, t_wu])
    hum_n  = np.concatenate([h_base, h_wu])
    snd_n  = np.concatenate([s_base, s_wu])
    y_n    = np.zeros(n_normal, dtype=int)

    # anomaly 1: overheating
    temp_a1 = rng.uniform(85.0, 102.0, n1)
    hum_a1  = rng.normal(50.0, 6.0, n1)
    snd_a1  = rng.normal(65.0, 4.0, n1)

    # anomaly 2: humidity surge
    temp_a2 = rng.normal(70.0, 5.0, n2)
    hum_a2  = rng.uniform(72.0, 88.0, n2)
    snd_a2  = rng.normal(65.0, 4.0, n2)

    # anomaly 3: sound spike
    temp_a3 = rng.normal(70.0, 5.0, n3)
    hum_a3  = rng.normal(50.0, 6.0, n3)
    snd_a3  = rng.uniform(76.0, 94.0, n3)

    # anomaly 4: all three sensors mildly elevated (harder case)
    temp_a4 = rng.normal(80.0, 4.5, n4)
    hum_a4  = rng.normal(64.0, 4.5, n4)
    snd_a4  = rng.normal(74.0, 4.5, n4)

    # anomaly 5: borderline, frequently missed
    temp_a5 = rng.normal(75.0, 3.0, n5)
    hum_a5  = rng.normal(60.0, 4.0, n5)
    snd_a5  = rng.normal(71.0, 3.0, n5)

    temp_a = np.concatenate([temp_a1, temp_a2, temp_a3, temp_a4, temp_a5])
    hum_a  = np.concatenate([hum_a1,  hum_a2,  hum_a3,  hum_a4,  hum_a5])
    snd_a  = np.concatenate([snd_a1,  snd_a2,  snd_a3,  snd_a4,  snd_a5])
    y_a    = np.ones(n_anomaly, dtype=int)

    # combine + shared sensor noise
    temperature  = np.concatenate([temp_n, temp_a])
    humidity     = np.concatenate([hum_n,  hum_a])
    sound_volume = np.concatenate([snd_n,  snd_a])
    is_anomaly   = np.concatenate([y_n,    y_a])

    temperature  += rng.normal(0, 1.2, n)
    humidity     += rng.normal(0, 1.2, n)
    sound_volume += rng.normal(0, 0.8, n)

    # shuffle and clip to physically plausible values
    idx = rng.permutation(n)
    df = pd.DataFrame({
        "temperature":  np.clip(temperature[idx],  20.0, 130.0).round(2),
        "humidity":     np.clip(humidity[idx],      0.0, 100.0).round(2),
        "sound_volume": np.clip(sound_volume[idx], 30.0, 130.0).round(2),
        "is_anomaly":   is_anomaly[idx],
    })
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 60 / 15 / 25 deterministic split (data already shuffled above).
    # - train: IsolationForest is fit on this part only (unsupervised).
    # - val:   used for threshold tuning via F1 sweep.
    # - test:  untouched until final evaluation.
    n       = len(df)
    n_train = int(n * 0.60)
    n_val   = int(n * 0.15)
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train: n_train + n_val].copy(),
        df.iloc[n_train + n_val:].copy(),
    )


# --- 2. training + threshold tuning -------------------------------------------

def train_model(X_train: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def tune_threshold(model: IsolationForest,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
    # sweep 400 candidate thresholds, keep the one with best F1 on val
    scores = model.score_samples(X_val)
    thresholds = np.linspace(scores.min(), scores.max(), 400)
    best_f1, best_thr = -1.0, float(np.percentile(scores, ANOMALY_RATE * 100))
    for thr in thresholds:
        preds = (scores < thr).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"  Threshold tuned on validation: {best_thr:.6f}  (val F1 = {best_f1:.4f})")
    return best_thr


def predict_with_threshold(model: IsolationForest, X: np.ndarray,
                           threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = model.score_samples(X)
    preds  = (scores < threshold).astype(int)
    return preds, scores


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             scores: np.ndarray, label: str) -> dict:
    result: dict = {
        "split":     label,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    try:
        result["roc_auc"] = round(roc_auc_score(y_true, -scores), 4)
    except Exception:
        result["roc_auc"] = None
    return result


# --- 3. plots -----------------------------------------------------------------

PALETTE = {
    "blue":   "#2c5f8a",
    "red":    "#e05555",
    "orange": "#e07b39",
    "green":  "#2c8a5f",
    "purple": "#8a2c5f",
}


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")


def save_metrics_table(metrics_list: list[dict]) -> None:
    cols       = ["split", "accuracy", "precision", "recall", "f1", "roc_auc"]
    col_labels = ["Split", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows       = [[m[c] for c in cols] for m in metrics_list]

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.2)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(PALETTE["blue"])
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        shade = "#e8f4fd" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(shade)
    ax.set_title(
        "Model Evaluation Metrics -- Model-to-Production IoT Anomalies\n"
        "(synthetic data with noise and borderline cases; "
        "imperfect metrics are expected)",
        fontsize=11, fontweight="bold", pad=16,
    )
    _save(fig, "metrics_table.png")


def save_metrics_csv(metrics_list: list[dict]) -> Path:
    path = OUTPUT_DIR / "metrics_table.csv"
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    print("  [saved] metrics_table.csv")
    return path


def _load_metrics_from_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict = {}
    for _, row in df.iterrows():
        key = str(row.get("split", "")).lower()
        if key:
            out[key] = {
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "roc_auc": None if pd.isna(row["roc_auc"]) else float(row["roc_auc"]),
            }
    return out


def write_model_metadata(threshold: float, n_train_samples: int, metrics_csv_path: Path) -> None:
    training_ts = datetime.now(timezone.utc).isoformat()
    metadata = {
        "project_name": PROJECT_NAME,
        "model_type": "IsolationForest",
        "model_version": MODEL_VERSION,
        "model_artifact_path": "models/iot_anomaly_iforest.pkl",
        "feature_names": FEATURES,
        "features": FEATURES,
        "contamination": CONTAMINATION,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_SEED,
        "threshold": round(threshold, 6),
        "training_script": "app/train_eval.py",
        "training_date_utc": training_ts,
        "training_date": training_ts,
        "n_train_samples": n_train_samples,
        "data_type": DATA_TYPE,
        "scenario": SCENARIO,
        "train_validation_test_split": SPLIT,
        "evaluation_metrics": _load_metrics_from_csv(metrics_csv_path),
        "limitation_note": (
            "The model was trained and evaluated on synthetic sensor data. "
            "Results validate the prototype workflow but do not prove real-world production performance."
        ),
    }
    with (META_DIR / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def write_dataset_metadata() -> None:
    metadata = {
        "project_name": PROJECT_NAME,
        "data_source": "fictional synthetic sensor data",
        "reason_for_synthetic_data": (
            "No real factory dataset was available in the module scope, so synthetic data was generated "
            "to demonstrate an end-to-end model-to-production workflow."
        ),
        "scenario": SCENARIO,
        "generated_by": "app/train_eval.py",
        "data_type": DATA_TYPE,
        "features": {
            "temperature": {
                "unit": "Celsius",
                "description": "Process temperature signal from the manufacturing line.",
            },
            "humidity": {
                "unit": "percent",
                "description": "Ambient or process humidity level near production equipment.",
            },
            "sound_volume": {
                "unit": "arbitrary sound level / normalized sound intensity",
                "description": "Acoustic intensity used as a proxy for vibration or mechanical stress.",
            },
        },
        "target_label": {
            "name": "is_anomaly",
            "description": (
                "Binary reference label used for offline evaluation only; it is not required as API input."
            ),
        },
        "noise_design": {
            "normal_variation": True,
            "borderline_cases": True,
            "overlapping_values": True,
            "injected_anomalies": True,
        },
        "split": SPLIT,
        "limitations": [
            "No real sensor hardware measurements are used.",
            "Synthetic distributions approximate but do not replicate production behavior.",
            "No long-term drift history is represented.",
        ],
    }
    with (META_DIR / "dataset_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def save_learning_dashboard(model: IsolationForest, threshold: float,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val:   np.ndarray, y_val:   np.ndarray,
                             X_test:  np.ndarray, y_test:  np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Dashboard -- Model-to-Production IoT Anomalies",
                 fontsize=14, fontweight="bold")

    # top-left: ROC curve (test set)
    _, scores_test = predict_with_threshold(model, X_test, threshold)
    fpr, tpr, _    = roc_curve(y_test, -scores_test)
    auc_val        = roc_auc_score(y_test, -scores_test)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=PALETTE["blue"], lw=2.5, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    # top-right: F1 vs decision threshold
    thresholds_sweep = np.linspace(scores_test.min(), scores_test.max(), 300)
    f1s = [f1_score(y_test, (scores_test < t).astype(int), zero_division=0)
           for t in thresholds_sweep]
    ax = axes[0, 1]
    ax.plot(-thresholds_sweep, f1s, color=PALETTE["orange"], lw=2.5)
    ax.axvline(x=-threshold, color=PALETTE["blue"], linestyle="--", lw=1.8,
               label=f"Tuned threshold ({-threshold:.3f})")
    ax.set_xlabel("Decision Score (negated, higher = more anomalous)")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs. Decision Threshold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # bottom-left: val F1 at different training-data fractions
    fracs    = np.linspace(0.10, 1.0, 9)
    f1_fracs = []
    for frac in fracs:
        n = max(100, int(len(X_train) * frac))
        m = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
                            random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X_train[:n])
        thr_f = tune_threshold(m, X_val, y_val) if len(X_val) > 0 else threshold
        pv, _ = predict_with_threshold(m, X_val, thr_f)
        f1_fracs.append(f1_score(y_val, pv, zero_division=0))
    ax = axes[1, 0]
    ax.plot(fracs * 100, f1_fracs, marker="o", color=PALETTE["green"], lw=2.5, markersize=7)
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Validation F1 Score")
    ax.set_title("Val F1 vs. Training Data Fraction")
    ax.grid(alpha=0.3)

    # bottom-right: val F1 at different n_estimators
    n_est_grid = [10, 25, 50, 75, 100, 150, 200]
    f1_ests    = []
    for n_est in n_est_grid:
        m = IsolationForest(n_estimators=n_est, contamination=CONTAMINATION,
                            random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X_train)
        thr_e = tune_threshold(m, X_val, y_val) if len(X_val) > 0 else threshold
        pv, _ = predict_with_threshold(m, X_val, thr_e)
        f1_ests.append(f1_score(y_val, pv, zero_division=0))
    ax = axes[1, 1]
    ax.plot(n_est_grid, f1_ests, marker="s", color=PALETTE["purple"], lw=2.5, markersize=7)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Validation F1 Score")
    ax.set_title("Val F1 vs. n_estimators")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "learning_dashboard_2x2.png")


def save_heatmaps_grid(X_train_df: pd.DataFrame,
                       y_val: np.ndarray, val_pred: np.ndarray,
                       y_test: np.ndarray, test_pred: np.ndarray) -> None:
    fig = plt.figure(figsize=(17, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)
    fig.suptitle("Feature Correlation and Confusion Matrices -- Model-to-Production IoT Anomalies",
                 fontsize=12, fontweight="bold")

    corr = X_train_df.corr()
    ax0 = fig.add_subplot(gs[0])
    im  = ax0.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    labs = corr.columns.tolist()
    ax0.set_xticks(range(len(labs)))
    ax0.set_yticks(range(len(labs)))
    ax0.set_xticklabels(labs, rotation=30, ha="right", fontsize=10)
    ax0.set_yticklabels(labs, fontsize=10)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax0.text(j, i, f"{corr.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title("Feature Correlation (Training Data)", fontsize=11)

    for col_idx, (yt, yp, lbl) in enumerate(
        [(y_val, val_pred, "Validation"), (y_test, test_pred, "Test")], start=1
    ):
        cm = confusion_matrix(yt, yp)
        ax = fig.add_subplot(gs[col_idx])
        im2 = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"], fontsize=10)
        ax.set_yticklabels(["Normal", "Anomaly"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"Confusion Matrix ({lbl})", fontsize=11)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

    _save(fig, "heatmaps_grid.png")


def save_histograms(df: pd.DataFrame) -> None:
    units = {"temperature": "C", "humidity": "%", "sound_volume": "dB"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Sensor Feature Distributions: Normal vs. Anomaly\n"
                 "Model-to-Production IoT Anomalies (synthetic data)",
                 fontsize=12, fontweight="bold")
    for ax, feat in zip(axes, FEATURES):
        norm = df.loc[df["is_anomaly"] == 0, feat]
        anom = df.loc[df["is_anomaly"] == 1, feat]
        ax.hist(norm, bins=50, alpha=0.65, color=PALETTE["blue"], label="Normal",  density=True)
        ax.hist(anom, bins=25, alpha=0.70, color=PALETTE["red"],  label="Anomaly", density=True)
        ax.set_xlabel(f"{feat} ({units[feat]})", fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Distribution: {feat}", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "histograms.png")


# --- main ---------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Model-to-Production IoT Anomalies -- Training Pipeline")
    print("  IU Module DLBDSMTP01")
    print("=" * 65)

    # 1. simulate data
    print("\n[1/6] Simulating sensor data ...")
    df = simulate_sensor_data()
    train, val, test = split_data(df)
    print(f"  Samples  : {len(df):,}  (train {len(train):,} / val {len(val):,} / test {len(test):,})")
    print(f"  Anomaly rate (injected) : {df['is_anomaly'].mean():.1%}")
    train.to_csv(DATA_DIR / "train.csv", index=False)
    val.to_csv(DATA_DIR / "val.csv",   index=False)
    test.to_csv(DATA_DIR / "test.csv",  index=False)
    print("  Saved -> data/train.csv, val.csv, test.csv")

    X_train = train[FEATURES].values;  y_train = train["is_anomaly"].values
    X_val   = val[FEATURES].values;    y_val   = val["is_anomaly"].values
    X_test  = test[FEATURES].values;   y_test  = test["is_anomaly"].values

    # 2. train
    print(f"\n[2/6] Training IsolationForest ...")
    print(f"  contamination={CONTAMINATION}  n_estimators={N_ESTIMATORS}  seed={RANDOM_SEED}")
    model = train_model(X_train)

    # 3. tune threshold
    print("\n[3/6] Tuning decision threshold on validation set ...")
    threshold = tune_threshold(model, X_val, y_val)

    # 4. evaluate
    print("\n[4/6] Evaluating ...")
    train_pred, train_scores = predict_with_threshold(model, X_train, threshold)
    val_pred,   val_scores   = predict_with_threshold(model, X_val,   threshold)
    test_pred,  test_scores  = predict_with_threshold(model, X_test,  threshold)

    metrics_list = [
        evaluate(y_train, train_pred, train_scores, "Train"),
        evaluate(y_val,   val_pred,   val_scores,   "Validation"),
        evaluate(y_test,  test_pred,  test_scores,  "Test"),
    ]
    hdr = f"  {'Split':12s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'AUC':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in metrics_list:
        print(
            f"  {m['split']:12s} | {m['accuracy']:6.4f} | {m['precision']:6.4f} | "
            f"{m['recall']:6.4f} | {m['f1']:6.4f} | {str(m['roc_auc']):>6s}"
        )

    # 5. save model + metadata
    print("\n[5/6] Saving model artifact ...")
    joblib.dump(model, MODEL_DIR / "iot_anomaly_iforest.pkl")
    metrics_csv_path = save_metrics_csv(metrics_list)
    write_model_metadata(threshold, int(len(X_train)), metrics_csv_path)
    write_dataset_metadata()
    print("  Saved -> models/iot_anomaly_iforest.pkl")
    print("  Saved -> metadata/model_metadata.json")
    print("  Saved -> metadata/dataset_metadata.json")

    # 6. plots
    print("\n[6/6] Generating visualisations ...")
    save_metrics_table(metrics_list)
    save_learning_dashboard(model, threshold,
                             X_train, y_train, X_val, y_val, X_test, y_test)
    save_heatmaps_grid(train[FEATURES], y_val, val_pred, y_test, test_pred)
    save_histograms(df)

    print("\n" + "=" * 65)
    print("  Training pipeline complete.  All outputs -> app/outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
