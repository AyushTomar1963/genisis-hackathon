    # src/train.py
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from src.data_pipeline import load_and_preprocess_data, engineer_features

# ---------------------- config ----------------------
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
TARGET_COL = "is_congested"     # change when real label available
TIME_COL = "timestamp"
N_SPLITS = 3
RANDOM_STATE = 42

# ---------------------- logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train")


def _dummy_data(n=180, d=6, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X += np.linspace(0, 1, n).reshape(-1, 1)  # mild drift
    y = rng.integers(0, 2, size=n)
    return X, y


def _select_Xy(df):
    # keep only numeric features; drop obvious non-features
    drop = {TARGET_COL, TIME_COL, "device_name", "link_id", "interface"}
    numeric_cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError("No numeric features found. Add engineered features or adjust drops.")
    X = df[numeric_cols].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()
    return X, y, numeric_cols


def train_model():
    log.info("Loading data...")
    df = load_and_preprocess_data()
    df = engineer_features(df)

    used_dummy = False
    if df is None:
        log.warning("No dataframe from pipeline; using dummy data.")
        X, y = _dummy_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        used_dummy = True
    else:
        # create a synthetic target if not present (structure demo)
        if TARGET_COL not in df.columns:
            log.warning("Target '%s' missing. Creating synthetic target from utilization > 0.8.", TARGET_COL)
            df[TARGET_COL] = (df.get("bandwidth_utilization", 0) > 0.8).astype(int)
        if TIME_COL in df.columns:
            df = df.sort_values(TIME_COL)
        X, y, feature_names = _select_Xy(df)

    # model + time-series CV
    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    accs, f1s = [], []
    log.info("Starting TimeSeriesSplit (%d folds)...", N_SPLITS)
    for i, (tr, te) in enumerate(tscv.split(X), start=1):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, average="binary")
        accs.append(acc); f1s.append(f1)
        log.info("Fold %d | acc: %.3f | f1: %.3f", i, acc, f1)

    log.info("CV mean | acc: %.3f | f1: %.3f", np.mean(accs), np.mean(f1s))

    # fit on full data and save
    clf.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "netsense_rf.pkl"
    joblib.dump({"model": clf, "features": feature_names, "used_dummy": used_dummy}, out_path)
    log.info("Saved model -> %s", out_path)


if __name__ == "__main__":
    train_model()
