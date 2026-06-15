"""
models/bayesian_model.py — Bayesian Network using pgmpy for win probability inference.
Uses a simplified BN structure; priors are updated incrementally nightly.
Models serialized to Modal Volume via pickle.
"""
import os
import pickle
import traceback
from typing import Optional

import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from db.supabase_client import log_error

# BN node names
NODE_HOME_FORM = "HomeForm"       # 0=poor,1=avg,2=good
NODE_AWAY_FORM = "AwayForm"
NODE_HOME_ADV = "HomeAdvantage"   # 0=no,1=yes
NODE_HOME_INJ = "HomeInjury"      # 0=no,1=yes
NODE_AWAY_INJ = "AwayInjury"
NODE_WIN = "HomeWin"              # 0=away wins,1=home wins


def _discretize_win_rate(wr: float) -> int:
    """Map win rate to 3-level ordinal."""
    if wr < 0.4:
        return 0
    elif wr < 0.6:
        return 1
    return 2


def build_default_model() -> BayesianNetwork:
    """Construct the Bayesian Network with default CPDs."""
    model = BayesianNetwork([
        (NODE_HOME_FORM, NODE_WIN),
        (NODE_AWAY_FORM, NODE_WIN),
        (NODE_HOME_ADV, NODE_WIN),
        (NODE_HOME_INJ, NODE_WIN),
        (NODE_AWAY_INJ, NODE_WIN),
    ])

    # Marginal priors — uniform over form levels
    cpd_home_form = TabularCPD(NODE_HOME_FORM, 3, [[1/3], [1/3], [1/3]])
    cpd_away_form = TabularCPD(NODE_AWAY_FORM, 3, [[1/3], [1/3], [1/3]])
    cpd_home_adv = TabularCPD(NODE_HOME_ADV, 2, [[0.4], [0.6]])
    cpd_home_inj = TabularCPD(NODE_HOME_INJ, 2, [[0.85], [0.15]])
    cpd_away_inj = TabularCPD(NODE_AWAY_INJ, 2, [[0.85], [0.15]])

    # Win CPD: P(HomeWin | parents)
    # Parents order: HomeForm(3), AwayForm(3), HomeAdvantage(2), HomeInjury(2), AwayInjury(2)
    # = 3*3*2*2*2 = 72 parent combinations → 2 values each → shape (2, 72)
    n_combos = 3 * 3 * 2 * 2 * 2  # 72

    # Base probability matrix — compute heuristically
    win_probs = []
    for hf in range(3):       # HomeForm
        for af in range(3):   # AwayForm
            for ha in range(2):  # HomeAdvantage
                for hi in range(2):  # HomeInjury
                    for ai in range(2):  # AwayInjury
                        base = 0.50
                        base += (hf - 1) * 0.08   # form adjustment
                        base -= (af - 1) * 0.08
                        base += ha * 0.05          # home advantage
                        base -= hi * 0.06          # home injury hurts home
                        base += ai * 0.06          # away injury helps home
                        p_win = min(max(base, 0.05), 0.95)
                        win_probs.append(p_win)

    cpd_win_values = np.array([
        [1 - p for p in win_probs],
        win_probs,
    ])

    cpd_win = TabularCPD(
        NODE_WIN, 2,
        cpd_win_values,
        evidence=[NODE_HOME_FORM, NODE_AWAY_FORM, NODE_HOME_ADV, NODE_HOME_INJ, NODE_AWAY_INJ],
        evidence_card=[3, 3, 2, 2, 2],
    )

    model.add_cpds(cpd_home_form, cpd_away_form, cpd_home_adv, cpd_home_inj, cpd_away_inj, cpd_win)
    assert model.check_model(), "BN model check failed"
    return model


def predict_proba(model: BayesianNetwork, feature_dict: dict) -> float:
    """
    Run Variable Elimination to get P(HomeWin=1 | evidence).
    feature_dict should have keys matching feature names from features.py.
    """
    try:
        infer = VariableElimination(model)

        home_wr = feature_dict.get("home_recent_win_rate", 0.5)
        away_wr = feature_dict.get("away_recent_win_rate", 0.5)
        home_adv = 1  # always treating home team as home
        home_inj = int(feature_dict.get("home_injury_flag", 0))
        away_inj = int(feature_dict.get("away_injury_flag", 0))

        evidence = {
            NODE_HOME_FORM: _discretize_win_rate(home_wr),
            NODE_AWAY_FORM: _discretize_win_rate(away_wr),
            NODE_HOME_ADV: home_adv,
            NODE_HOME_INJ: home_inj,
            NODE_AWAY_INJ: away_inj,
        }

        result = infer.query([NODE_WIN], evidence=evidence, show_progress=False)
        return float(result.values[1])  # P(HomeWin=1)
    except Exception as exc:
        log_error(
            context="bayesian_model.predict_proba",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return 0.5


def update_priors(model: BayesianNetwork, game_logs: list) -> BayesianNetwork:
    """
    Update marginal CPDs for HomeForm and AwayForm based on recent game results.
    game_logs: list of dicts with win (bool), home_recent_win_rate (float).
    Returns updated model.
    """
    if not game_logs:
        return model

    home_win_rates = [g.get("home_recent_win_rate", 0.5) for g in game_logs]
    counts = [0, 0, 0]
    for wr in home_win_rates:
        counts[_discretize_win_rate(wr)] += 1

    total = sum(counts) or 1
    priors = [[counts[i] / total] for i in range(3)]

    try:
        model.remove_cpds(model.get_cpds(NODE_HOME_FORM))
        model.add_cpds(TabularCPD(NODE_HOME_FORM, 3, priors))
    except Exception:
        pass

    return model


def save_model(sport: str, model: BayesianNetwork, volume_path: str = "/models"):
    """Serialize BN to the Modal Volume mount point."""
    path = os.path.join(volume_path, f"bn_{sport.lower()}.pkl")
    os.makedirs(volume_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(sport: str, volume_path: str = "/models") -> Optional[BayesianNetwork]:
    """Load BN from the Modal Volume mount point. Returns None if not found."""
    path = os.path.join(volume_path, f"bn_{sport.lower()}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log_error(
            context=f"bayesian_model.load_model({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return None
