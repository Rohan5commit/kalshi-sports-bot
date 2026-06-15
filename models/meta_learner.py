"""
models/meta_learner.py — Logistic regression meta-learner that combines
XGBoost and Bayesian Network probabilities into a single stacked estimate.
"""
import os
import pickle
import traceback
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from db.supabase_client import log_error


def train_meta_learner(
    xgb_probs: np.ndarray,
    bn_probs: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """
    Train the meta-learner on stacked probability outputs.
    xgb_probs, bn_probs: 1-D arrays of home-win probability from each model.
    labels: 1-D array of 0/1 ground truth.
    """
    X_meta = np.column_stack([xgb_probs, bn_probs])
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_meta, labels)
    return clf


def predict(
    meta_model: LogisticRegression,
    xgb_prob: float,
    bn_prob: float,
) -> float:
    """
    Return the stacked probability that the home team wins.
    """
    X = np.array([[xgb_prob, bn_prob]])
    try:
        return float(meta_model.predict_proba(X)[0, 1])
    except Exception as exc:
        log_error(
            context="meta_learner.predict",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return (xgb_prob + bn_prob) / 2.0


def save_model(sport: str, model: LogisticRegression, volume_path: str = "/models"):
    """Serialize to Modal Volume."""
    path = os.path.join(volume_path, f"meta_{sport.lower()}.pkl")
    os.makedirs(volume_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(sport: str, volume_path: str = "/models") -> Optional[LogisticRegression]:
    """Load from Modal Volume. Returns None if not found."""
    path = os.path.join(volume_path, f"meta_{sport.lower()}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log_error(
            context=f"meta_learner.load_model({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return None
