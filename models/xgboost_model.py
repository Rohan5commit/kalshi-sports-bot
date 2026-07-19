"""
models/xgboost_model.py — XGBoost classifier for game outcome prediction.
Supports full training from historical data and incremental retraining on new results.
Models are persisted to / loaded from Modal Volume (kalshi-bot-models).
"""
import os
import io
import pickle
import traceback
from typing import Optional

import numpy as np
import xgboost as xgb

from config import MODEL_VOLUME_NAME, SPORTS
from data.features import SPORT_FEATURES, build_training_dataset
from db.supabase_client import log_error

MODEL_PATH_TMPL = "/models/xgb_{sport}.pkl"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
    "tree_method": "hist",
    "random_state": 42,
}

INCREMENTAL_ROUNDS = 20  # number of boosting rounds for incremental update


def _model_path(sport: str) -> str:
    return MODEL_PATH_TMPL.format(sport=sport.lower())


def train(sport: str, game_logs: list) -> xgb.XGBClassifier:
    """
    Full training from historical game logs.
    Returns a fitted XGBClassifier.
    """
    X, y = build_training_dataset(sport, game_logs)
    if X.shape[0] == 0:
        raise ValueError(f"No training data for {sport}")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)
    return model


def incremental_retrain(sport: str, new_game_logs: list, model: xgb.XGBClassifier) -> xgb.XGBClassifier:
    """
    Update an existing XGBoost model with new game results without full refit.
    Uses xgb.train() continuation via get_booster().
    """
    X_new, y_new = build_training_dataset(sport, new_game_logs)
    if X_new.shape[0] == 0:
        return model

    dmatrix = xgb.DMatrix(X_new, label=y_new)
    booster = model.get_booster()
    updated_booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": XGB_PARAMS["max_depth"],
            "learning_rate": XGB_PARAMS["learning_rate"],
            "subsample": XGB_PARAMS["subsample"],
            "colsample_bytree": XGB_PARAMS["colsample_bytree"],
            "tree_method": XGB_PARAMS["tree_method"],
        },
        dmatrix,
        num_boost_round=INCREMENTAL_ROUNDS,
        xgb_model=booster,
    )

    model._Booster = updated_booster
    return model


def predict_proba(model: xgb.XGBClassifier, feature_vector: np.ndarray) -> float:
    """
    Return the probability that the home team wins.
    feature_vector: 1-D numpy array.
    """
    X = feature_vector.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    return float(proba[1])


def save_model(sport: str, model: xgb.XGBClassifier, volume_path: str = "/models"):
    """Serialize model to the Modal Volume mount point."""
    path = os.path.join(volume_path, f"xgb_{sport.lower()}.pkl")
    os.makedirs(volume_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(sport: str, volume_path: str = "/models") -> Optional[xgb.XGBClassifier]:
    """Load model from the Modal Volume mount point. Returns None if not found."""
    path = os.path.join(volume_path, f"xgb_{sport.lower()}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log_error(
            context=f"xgboost_model.load_model({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return None


def compute_rolling_accuracy(sport: str, predictions_with_outcomes: list, window: int = 14) -> float:
    """
    Compute rolling accuracy over the last `window` days.
    predictions_with_outcomes: list of dicts with keys: final_prob, actual_win (bool).
    Returns accuracy as a float in [0, 1].
    """
    if not predictions_with_outcomes:
        return 0.5

    recent = predictions_with_outcomes[-window:]
    correct = 0
    for p in recent:
        predicted_win = p.get("final_prob", 0.5) >= 0.5
        actual_win = p.get("actual_win", False)
        if predicted_win == actual_win:
            correct += 1

    return correct / len(recent) if recent else 0.5

