"""
models/calibration.py — Isotonic regression calibration for the stacked model output.
Calibrates raw logistic regression probabilities to be better-aligned with true frequencies.
"""
import os
import pickle
import traceback
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from db.supabase_client import log_error


def train_calibrator(raw_probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator.
    raw_probs: 1-D array of meta-learner probabilities.
    labels: 1-D binary array of ground truth.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, labels)
    return calibrator


def calibrate(calibrator: IsotonicRegression, raw_prob: float) -> float:
    """
    Apply calibration to a single probability.
    """
    try:
        return float(calibrator.predict([raw_prob])[0])
    except Exception as exc:
        log_error(
            context="calibration.calibrate",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return raw_prob


def calibrate_batch(calibrator: IsotonicRegression, raw_probs: np.ndarray) -> np.ndarray:
    """Apply calibration to an array of probabilities."""
    try:
        return calibrator.predict(raw_probs)
    except Exception as exc:
        log_error(
            context="calibration.calibrate_batch",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return raw_probs


def save_model(sport: str, calibrator: IsotonicRegression, volume_path: str = "/models"):
    """Serialize to Modal Volume."""
    path = os.path.join(volume_path, f"calib_{sport.lower()}.pkl")
    os.makedirs(volume_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_model(sport: str, volume_path: str = "/models") -> Optional[IsotonicRegression]:
    """Load from Modal Volume. Returns None if not found."""
    path = os.path.join(volume_path, f"calib_{sport.lower()}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log_error(
            context=f"calibration.load_model({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return None
