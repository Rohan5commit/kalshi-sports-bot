"""
models/bayesian_model.py — Bayesian Network using pgmpy for win probability inference.
Uses DiscreteBayesianNetwork (pgmpy >= 1.1) with fallback to BayesianNetwork for older versions.
"""
import os
import pickle
import traceback
from typing import Optional

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from db.supabase_client import log_error

# pgmpy >= 1.1 renamed BayesianNetwork to DiscreteBayesianNetwork
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

NODE_HOME_FORM = "HomeForm"
NODE_AWAY_FORM = "AwayForm"
NODE_HOME_ADV  = "HomeAdvantage"
NODE_HOME_INJ  = "HomeInjury"
NODE_AWAY_INJ  = "AwayInjury"
NODE_WIN       = "HomeWin"


def _discretize_win_rate(wr: float) -> int:
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
        (NODE_HOME_ADV,  NODE_WIN),
        (NODE_HOME_INJ,  NODE_WIN),
        (NODE_AWAY_INJ,  NODE_WIN),
    ])

    cpd_home_form = TabularCPD(NODE_HOME_FORM, 3, [[1/3], [1/3], [1/3]])
    cpd_away_form = TabularCPD(NODE_AWAY_FORM, 3, [[1/3], [1/3], [1/3]])
    cpd_home_adv  = TabularCPD(NODE_HOME_ADV,  2, [[0.4], [0.6]])
    cpd_home_inj  = TabularCPD(NODE_HOME_INJ,  2, [[0.85], [0.15]])
    cpd_away_inj  = TabularCPD(NODE_AWAY_INJ,  2, [[0.85], [0.15]])

    n_combos = 3 * 3 * 2 * 2 * 2  # 72
    win_probs = []
    for hf in range(3):
        for af in range(3):
            for ha in range(2):
                for hi in range(2):
                    for ai in range(2):
                        base = 0.50
                        base += (hf - 1) * 0.08
                        base -= (af - 1) * 0.08
                        base += ha * 0.05
                        base -= hi * 0.06
                        base += ai * 0.06
                        win_probs.append(min(max(base, 0.05), 0.95))

    cpd_win = TabularCPD(
        NODE_WIN, 2,
        [[1 - p for p in win_probs], win_probs],
        evidence=[NODE_HOME_FORM, NODE_AWAY_FORM, NODE_HOME_ADV, NODE_HOME_INJ, NODE_AWAY_INJ],
        evidence_card=[3, 3, 2, 2, 2],
    )

    model.add_cpds(cpd_home_form, cpd_away_form, cpd_home_adv, cpd_home_inj, cpd_away_inj, cpd_win)
    assert model.check_model(), "BN model check failed"
    return model


def predict_proba(model: BayesianNetwork, feature_dict: dict) -> float:
    """Run Variable Elimination to get P(HomeWin=1 | evidence)."""
    try:
        infer = VariableElimination(model)
        evidence = {
            NODE_HOME_FORM: _discretize_win_rate(feature_dict.get("home_recent_win_rate", 0.5)),
            NODE_AWAY_FORM: _discretize_win_rate(feature_dict.get("away_recent_win_rate", 0.5)),
            NODE_HOME_ADV:  1,
            NODE_HOME_INJ:  int(feature_dict.get("home_injury_flag", 0)),
            NODE_AWAY_INJ:  int(feature_dict.get("away_injury_flag", 0)),
        }
        result = infer.query([NODE_WIN], evidence=evidence, show_progress=False)
        return float(result.values[1])
    except Exception as exc:
        log_error(
            context="bayesian_model.predict_proba",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return 0.5


def update_priors(model: BayesianNetwork, game_logs: list) -> BayesianNetwork:
    """Update HomeForm marginal CPD from recent game results."""
    if not game_logs:
        return model
    counts = [0, 0, 0]
    for g in game_logs:
        counts[_discretize_win_rate(g.get("home_recent_win_rate", 0.5))] += 1
    total = sum(counts) or 1
    priors = [[counts[i] / total] for i in range(3)]
    try:
        model.remove_cpds(model.get_cpds(NODE_HOME_FORM))
        model.add_cpds(TabularCPD(NODE_HOME_FORM, 3, priors))
    except Exception:
        pass
    return model


def save_model(sport: str, model: BayesianNetwork, volume_path: str = "/models"):
    path = os.path.join(volume_path, f"bn_{sport.lower()}.pkl")
    os.makedirs(volume_path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(sport: str, volume_path: str = "/models") -> Optional[BayesianNetwork]:
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
