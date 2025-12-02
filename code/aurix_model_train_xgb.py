
# aurix_model_train_xgb.py
# Upgraded training module for AURIX-AI using XGBoost with early stopping,
# profit-aware thresholding, and two-model uplift with XGB classifiers.
# CPU-friendly defaults (tree_method='hist'); GPU can be enabled via env vars.

import json
import joblib
import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, List
from xgboost import XGBClassifier
from dataclasses import dataclass
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    log_loss,
)

# Ensure src/ is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = Path("/Users/mlwu/Documents/Academia/CMU/tepper_courses/Machine Learning for Business Applications/project/code/data/figs")
DATA_OUT_DIR = Path("/Users/mlwu/Documents/Academia/CMU/tepper_courses/Machine Learning for Business Applications/project/code/data/data_out")

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Disable plot titles
plt.rcParams["axes.titlepad"] = 0
plt.rcParams["axes.titlesize"] = 0

# Configuration for profit calculation
@dataclass
class ProfitConfig:
    margin_per_customer: float = 100.0
    contact_cost: float = 5.0
    max_contact_rate: float = 0.30      # fraction of population to contact (budget)
    uplift_effect_guess: float = 0.20   # used for risk-only profit curve baseline

# Load engineered features
def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load engineered features for churn modeling.

    By default, reads from data/aurix_features.csv, but can be overridden
    with the AURIX_FEATURES_PATH environment variable.
    """
    default_path = Path("data") / "aurix_features.csv"
    path_str = os.getenv("AURIX_FEATURES_PATH", str(default_path))

    print(f"[aurix] Loading features from: {path_str}")
    df = pd.read_csv(path_str)

    if "churn_flag" not in df.columns:
        raise ValueError(
            'Expected a "churn_flag" column in the engineered features dataset.'
        )

    y = df["churn_flag"]
    X = df.drop(columns=["churn_flag"])

    print(f"[aurix] Loaded features: X shape = {X.shape}, y length = {len(y)}")
    return X, y

def train_xgb(
    X,
    y,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train a baseline XGBoost churn classifier with an explicit
    train/validation/test split.

    We do not use callbacks or early_stopping_rounds here because the
    installed xgboost sklearn wrapper does not accept those arguments.
    """
    # Split into train+val vs test
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Then split tmp into train and validation
    val_fraction_of_tmp = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_fraction_of_tmp,
        random_state=random_state,
        stratify=y_tmp,
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,  # use all available cores
    )

    # No callbacks / no early_stopping_rounds: keep API-compatible
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    # Evaluate on held-out test set
    y_proba_te = model.predict_proba(X_te)[:, 1]
    auc_te = roc_auc_score(y_te, y_proba_te)
    print(f"[aurix] Test AUC (XGB churn): {auc_te:0.4f}")

    splits: Dict[str, Any] = {
        "X_tr": X_tr,
        "X_va": X_va,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_va": y_va,
        "y_te": y_te,
    }
    return model, splits

# Evaluate classifier performance with profit curve
def evaluate_classifier(
    model: XGBClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: ProfitConfig,
    out_prefix: str = "xgb"
):
    proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba)
    ll = log_loss(y_valid, proba, labels=[0, 1])

    y_pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary", zero_division=0)

    profit_curve = compute_profit_curve_risk(proba, y_valid.to_numpy(), cfg)
    plot_profit_curve(profit_curve, f"{out_prefix}_profit_curve.pdf")
    plot_roc(proba, y_valid.to_numpy(), f"{out_prefix}_roc.pdf")

    n = len(proba)
    k = int(cfg.max_contact_rate * n)
    top_idx = np.argsort(-proba)[:k]
    profit_opt_pred = np.zeros(n, dtype=int)
    profit_opt_pred[top_idx] = 1
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_valid, profit_opt_pred, average="binary", zero_division=0)

    metrics = {
        "auc": float(auc),
        "logloss": float(ll),
        "precision@0.5": float(prec),
        "recall@0.5": float(rec),
        "f1@0.5": float(f1),
        f"precision@top{int(cfg.max_contact_rate*100)}pct": float(prec_b),
        f"recall@top{int(cfg.max_contact_rate*100)}pct": float(rec_b),
        f"f1@top{int(cfg.max_contact_rate*100)}pct": float(f1_b),
    }
    return metrics, proba

# Compute profit curve based on risk scores
def compute_profit_curve_risk(scores: np.ndarray, y_true: np.ndarray, cfg: ProfitConfig) -> pd.DataFrame:
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    n = len(scores_sorted)
    ks = np.arange(1, int(cfg.max_contact_rate * n) + 1)

    profits = []
    for k in ks:
        s = scores_sorted[:k].sum()
        expected_retained = cfg.uplift_effect_guess * s
        profit = cfg.margin_per_customer * expected_retained - cfg.contact_cost * k
        profits.append(profit)

    return pd.DataFrame({"k": ks, "contact_rate": ks / n, "profit": profits})

# Plot profit curve
def plot_profit_curve(curve: pd.DataFrame, filename: str):
    outpath = FIG_DIR / filename
    plt.figure(figsize=(6, 4))
    plt.plot(curve["contact_rate"], curve["profit"])
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Contact Rate")
    plt.ylabel("Expected Profit")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# Plot ROC curve
def plot_roc(scores: np.ndarray, y_true: np.ndarray, filename: str):
    outpath = FIG_DIR / filename
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="XGB")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# Two-model uplift modeling with XGBoost
def two_model_uplift_xgb(
    X,
    y,
    treatment
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Two-model uplift using independent XGB classifiers.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Binary outcome (0/1).
    treatment : pd.Series or np.ndarray
        Binary treatment indicator (1 = treated, 0 = control).

    Returns
    -------
    uplift_scores : np.ndarray
        Estimated uplift per observation: P(Y=1 | T=1, X) - P(Y=1 | T=0, X).
    meta : dict
        Dictionary containing:
          - 'treated_model': fitted XGBClassifier on treated group
          - 'control_model': fitted XGBClassifier on control group
          - 'treated_preds': np.ndarray of treated predictions
          - 'control_preds': np.ndarray of control predictions
          - 'indices': dict of train/valid indices for each group
    """

    # Normalize X to DataFrame for consistent .iloc indexing
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_df = pd.DataFrame(X)

    # Normalize y to Series
    if isinstance(y, pd.Series):
        y_ser = y
    else:
        y_ser = pd.Series(y)

    # Normalize treatment to 1D numpy array
    if isinstance(treatment, (pd.Series, pd.DataFrame)):
        t_arr = np.asarray(treatment).reshape(-1)
    else:
        t_arr = np.asarray(treatment).reshape(-1)

    if len(y_ser) != len(X_df) or len(t_arr) != len(X_df):
        raise ValueError("X, y, and treatment must have the same number of rows.")

    # Index array over rows
    idx = np.arange(len(y_ser))

    # Boolean masks for treated/control
    t_mask = t_arr == 1
    c_mask = t_arr == 0

    # Global train/valid split indices
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y_ser  # optional, but usually good
    )

    # Group-specific train/valid index sets (row positions)
    t_tr = np.intersect1d(idx[t_mask], tr_idx)
    t_va = np.intersect1d(idx[t_mask], va_idx)
    c_tr = np.intersect1d(idx[c_mask], tr_idx)
    c_va = np.intersect1d(idx[c_mask], va_idx)

    def fit_split(
        X_frame: pd.DataFrame,
        y_series: pd.Series,
        tr_rows: np.ndarray,
        va_rows: np.ndarray
    ) -> XGBClassifier:
        """
        Fit an XGBClassifier on the given row indices.
        Validation indices are currently unused (no early stopping),
        but kept for future extension.
        """
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
        )

        # Use row-based indexing via .iloc
        model.fit(
            X_frame.iloc[tr_rows],
            y_series.iloc[tr_rows],
        )
        return model

    # Fit separate models on treated and control subsets
    mdl_t = fit_split(X_df, y_ser, t_tr, t_va)
    mdl_c = fit_split(X_df, y_ser, c_tr, c_va)

    # Predict probabilities on full population
    preds_t = mdl_t.predict_proba(X_df)[:, 1]
    preds_c = mdl_c.predict_proba(X_df)[:, 1]
    uplift_scores = preds_t - preds_c

    meta = {
        "treated_model": mdl_t,
        "control_model": mdl_c,
        "treated_preds": preds_t,
        "control_preds": preds_c,
        "indices": {
            "t_tr": t_tr,
            "t_va": t_va,
            "c_tr": c_tr,
            "c_va": c_va,
        },
    }

    return uplift_scores, meta

# # Two-model uplift modeling with XGBoost
# def two_model_uplift_xgb(
#     X,
#     y,
#     treatment
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """
#     Two-model uplift using independent XGB classifiers.

#     Parameters
#     ----------
#     X : pd.DataFrame or np.ndarray
#         Feature matrix.
#     y : pd.Series or np.ndarray
#         Binary outcome (0/1).
#     treatment : pd.Series or np.ndarray
#         Binary treatment indicator (1 = treated, 0 = control).

#     Returns
#     -------
#     uplift_scores : np.ndarray
#         Estimated uplift per observation: P(Y=1 | T=1, X) - P(Y=1 | T=0, X).
#     meta : dict
#         Dictionary containing:
#           - 'treated_model': fitted XGBClassifier on treated group
#           - 'control_model': fitted XGBClassifier on control group
#           - 'treated_preds': np.ndarray of treated predictions
#           - 'control_preds': np.ndarray of control predictions
#           - 'indices': dict of train/valid indices for each group
#     """

#     # Normalize X to DataFrame for consistent .iloc indexing
#     if isinstance(X, pd.DataFrame):
#         X_df = X
#     else:
#         X_df = pd.DataFrame(X)

#     # Normalize y to Series
#     if isinstance(y, pd.Series):
#         y_ser = y
#     else:
#         y_ser = pd.Series(y)

#     # Normalize treatment to 1D numpy array
#     if isinstance(treatment, (pd.Series, pd.DataFrame)):
#         t_arr = np.asarray(treatment).reshape(-1)
#     else:
#         t_arr = np.asarray(treatment).reshape(-1)

#     if len(y_ser) != len(X_df) or len(t_arr) != len(X_df):
#         raise ValueError("X, y, and treatment must have the same number of rows.")

#     # Index array over rows
#     idx = np.arange(len(y_ser))

#     # Boolean masks for treated/control
#     t_mask = t_arr == 1
#     c_mask = t_arr == 0

#     # Global train/valid split indices
#     tr_idx, va_idx = train_test_split(
#         idx,
#         test_size=0.2,
#         shuffle=True,
#         random_state=42,
#         stratify=y_ser  # optional, but usually good
#     )

#     # Group-specific train/valid index sets (row positions)
#     t_tr = np.intersect1d(idx[t_mask], tr_idx)
#     t_va = np.intersect1d(idx[t_mask], va_idx)
#     c_tr = np.intersect1d(idx[c_mask], tr_idx)
#     c_va = np.intersect1d(idx[c_mask], va_idx)

#     def fit_split(X_frame: pd.DataFrame,
#                   y_series: pd.Series,
#                   tr_rows: np.ndarray,
#                   va_rows: np.ndarray) -> XGBClassifier:
#         """
#         Fit an XGBClassifier on the given row indices.
#         Validation indices are currently unused (no early stopping),
#         but kept for future extension.
#         """
#         model = XGBClassifier(
#             n_estimators=400,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             tree_method="hist",
#             n_jobs=-1,
#             eval_metric="logloss",
#         )

#         # Use row-based indexing via .iloc
#         model.fit(
#             X_frame.iloc[tr_rows],
#             y_series.iloc[tr_rows]
#         )
#         return model

#     # Fit separate models on treated and control subsets
#     mdl_t = fit_split(X_df, y_ser, t_tr, t_va)
#     mdl_c = fit_split(X_df, y_ser, c_tr, c_va)

#     # Predict probabilities on full population
#     preds_t = mdl_t.predict_proba(X_df)[:, 1]
#     preds_c = mdl_c.predict_proba(X_df)[:, 1]
#     uplift_scores = preds_t - preds_c

#     meta = {
#         "treated_model": mdl_t,
#         "control_model": mdl_c,
#         "treated_preds": preds_t,
#         "control_preds": preds_c,
#         "indices": {
#             "t_tr": t_tr,
#             "t_va": t_va,
#             "c_tr": c_tr,
#             "c_va": c_va,
#         },
#     }

#     return uplift_scores, meta





# def two_model_uplift_xgb(
#     X: np.ndarray,
#     y: np.ndarray,
#     treatment: np.ndarray,
#     test_size: float = 0.2,
#     random_state: int = 42,
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """
#     Two-model uplift approach using XGBoost.

#     - Model T: trained only on treated customers (treatment == 1)
#     - Model C: trained only on control customers (treatment == 0)

#     Returns:
#         uplift_scores: pT(x) - pC(x) on a held-out test set
#         tm: diagnostic bundle with models and test split
#     """
#     # Ensure numpy arrays
#     X = np.asarray(X)
#     y = np.asarray(y)
#     t = np.asarray(treatment)

#     # Basic sanity checks
#     if X.shape[0] != y.shape[0] or X.shape[0] != t.shape[0]:
#         raise ValueError(
#             f"X, y, treatment must have same number of rows; got "
#             f"{X.shape[0]}, {y.shape[0]}, {t.shape[0]}"
#         )

#     # Split into train vs test for uplift evaluation
#     X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
#         X,
#         y,
#         t,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=t,
#     )

#     # Within the training portion, create a random train/val split
#     n_tr = X_tr.shape[0]
#     rng = np.random.RandomState(random_state)
#     base_mask = rng.rand(n_tr) < 0.8  # 80% train, 20% val

#     # Train / val masks by group
#     t_tr_mask = (t_tr == 1) & base_mask
#     t_va_mask = (t_tr == 1) & (~base_mask)

#     c_tr_mask = (t_tr == 0) & base_mask
#     c_va_mask = (t_tr == 0) & (~base_mask)

#     # Fit treated and control models using existing helper
#     mdl_t = fit_split(X_tr, y_tr, t_tr_mask, t_va_mask)
#     mdl_c = fit_split(X_tr, y_tr, c_tr_mask, c_va_mask)

#     # Predict probabilities on held-out test set
#     p_treat = mdl_t.predict_proba(X_te)[:, 1]
#     p_ctrl = mdl_c.predict_proba(X_te)[:, 1]

#     uplift_scores = p_treat - p_ctrl

#     # Simple diagnostic printout
#     print(
#         "[aurix] Uplift scores: "
#         f"min={uplift_scores.min():.4f}, "
#         f"max={uplift_scores.max():.4f}, "
#         f"mean={uplift_scores.mean():.4f}"
#     )

#     tm: Dict[str, Any] = {
#         "mdl_treated": mdl_t,
#         "mdl_control": mdl_c,
#         "X_test": X_te,
#         "y_test": y_te,
#         "t_test": t_te,
#         "uplift_scores": uplift_scores,
#     }

#     return uplift_scores, tm

# # Helper to fit model on subset
# def fit_split(
#     X: np.ndarray,
#     y: np.ndarray,
#     train_mask: np.ndarray,
#     val_mask: np.ndarray,
# )   -> XGBClassifier:
#     """
#     Helper for two-model uplift: fit an XGB model on a subset.
#     No callbacks / early stopping to remain compatible with the installed
#     XGBoost sklearn wrapper.
#     """
#     X_tr, y_tr = X[train_mask], y[train_mask]
#     X_va, y_va = X[val_mask], y[val_mask]

#     model = XGBClassifier(
#         n_estimators=400,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective="binary:logistic",
#         eval_metric="logloss",
#         tree_method="hist",
#         random_state=42,
#         n_jobs=-1,
#     )

#     model.fit(
#         X_tr,
#         y_tr,
#         eval_set=[(X_va, y_va)],
#         verbose=False,
#     )
#     return model
    
    # def fit_split(X_sub, y_sub, tr_mask, va_mask):
    #     pos = (y_sub[tr_mask] == 1).sum()
    #     neg = (y_sub[tr_mask] == 0).sum()
    #     spw = max((neg / max(pos, 1)), 1.0)
    #     params = dict(
    #         n_estimators=2000, max_depth=4, learning_rate=0.05,
    #         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    #         random_state=random_state, scale_pos_weight=spw,
    #         tree_method=os.getenv("AURIX_TREE_METHOD", "hist"),
    #         predictor=os.getenv("AURIX_PREDICTOR", "cpu_predictor"),
    #         n_jobs=-1,
    #     )
    #     model = XGBClassifier(**params)
    #     model.fit(
    #         X_sub[tr_mask], y_sub[tr_mask],
    #         eval_set=[(X_sub[va_mask], y_sub[va_mask])],
    #         verbose=False       # early_stopping_rounds=100
    #     )
    #     return model

    t_mask = (treatment == 1)
    X_t, y_t = X[t_mask], y[t_mask]
    t_tr_mask = np.isin(np.where(t_mask)[0], tr_idx)
    t_va_mask = np.isin(np.where(t_mask)[0], va_idx)
    mdl_t = fit_split(X_t.to_numpy(), y_t.to_numpy(), t_tr_mask, t_va_mask)

    c_mask = (treatment == 0)
    X_c, y_c = X[c_mask], y[c_mask]
    c_tr_mask = np.isin(np.where(c_mask)[0], tr_idx)
    c_va_mask = np.isin(np.where(c_mask)[0], va_idx)
    mdl_c = fit_split(X_c.to_numpy(), y_c.to_numpy(), c_tr_mask, c_va_mask)

    p_t = mdl_t.predict_proba(X.to_numpy())[:, 1]
    p_c = mdl_c.predict_proba(X.to_numpy())[:, 1]
    uplift = p_t - p_c

    return uplift, {"treated_model": mdl_t, "control_model": mdl_c}

# Qini curve computation
def qini_curve(uplift: np.ndarray, treatment: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-uplift)
    t_sorted = treatment[order]
    y_sorted = y_true[order]
    n = len(y_sorted)
    ks = np.arange(1, n + 1)
    inc = np.cumsum(2 * t_sorted * y_sorted - t_sorted - y_sorted)
    return pd.DataFrame({"k": ks, "rate": ks / n, "qini": inc})

# Plot Qini curve
def plot_qini(curve: pd.DataFrame, outpath: str = "qini_placeholder.pdf"):
    plt.figure(figsize=(6, 4))
    plt.plot(curve["rate"], curve["qini"])
    plt.axhline(0, linestyle="--")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Targeted Rate")
    plt.ylabel("Incremental Response (arb. units)")
    # plt.title("Qini Curve (Two-Model Uplift, XGB)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# Feature importance plot
def feature_importance_plot(model: XGBClassifier, feature_names: List[str], outpath: str = "xgb_feature_importance.pdf"):
    try:
        importance = model.feature_importances_
    except Exception:
        importance = None
    if importance is None:
        return
    idx = np.argsort(-importance)[:25]
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(feature_names)[idx][::-1], importance[idx][::-1])
    plt.xlabel("Importance (Gain)")
    # plt.title("Top Feature Importances (XGB)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# Main execution
def main():
    X, y = load_features()

    model, splits = train_xgb(X, y)
    X_train = splits["X_tr"]
    X_valid = splits["X_va"]
    y_train = splits["y_tr"]
    y_valid = splits["y_va"]

    joblib.dump(model, "aurix_churn_model_xgb.joblib")

    cfg = ProfitConfig(
        margin_per_customer=float(os.getenv("AURIX_MARGIN", 100.0)),
        contact_cost=float(os.getenv("AURIX_CONTACT_COST", 5.0)),
        max_contact_rate=float(os.getenv("AURIX_MAX_CONTACT_RATE", 0.3)),
        uplift_effect_guess=float(os.getenv("AURIX_UPLIFT_GUESS", 0.20)),
    )
    metrics, proba_valid = evaluate_classifier(model, X_valid, y_valid, cfg, out_prefix="xgb")
    with open("aurix_churn_eval_xgb.json", "w") as f:
        json.dump(metrics, f, indent=2)

    feature_importance_plot(model, X.columns.tolist())

    if "treatment" in X.columns:
        treatment = X["treatment"].to_numpy().astype(int)
        X_uplift = X.drop(columns=["treatment"])
    else:
        rng = np.random.RandomState(42)
        treatment = (rng.rand(len(y)) < 0.30).astype(int)
        X_uplift = X

    uplift_scores, tm = two_model_uplift_xgb(X_uplift, y, treatment)
    uplift_scores_path = DATA_OUT_DIR / "aurix_uplift_scores_xgb.npy"
    np.save(uplift_scores_path, uplift_scores)
    print(f"[aurix] Saved uplift scores to {uplift_scores_path}")

    # Minimal eval frame with uplift, treatment, and outcome
    uplift_eval = pd.DataFrame(
        {
            "uplift": uplift_scores,
            "treatment": np.asarray(treatment).reshape(-1),
            "y": np.asarray(y).reshape(-1),
        }
    )
    uplift_eval_path = DATA_OUT_DIR / "aurix_uplift_eval_xgb.csv"
    uplift_eval.to_csv(uplift_eval_path, index=False)
    print(f"[aurix] Saved uplift eval data to {uplift_eval_path}")
    
    qini = qini_curve(uplift_scores, treatment, y.to_numpy())
    plot_qini(qini, "qini_placeholder.pdf")

    # joblib.dump(tm["treated_model"], "aurix_uplift_model_treated_xgb.joblib")
    # joblib.dump(tm["control_model"], "aurix_uplift_model_control_xgb.joblib")
    joblib.dump(tm["treated_model"], DATA_OUT_DIR / "aurix_uplift_model_treated_xgb.joblib")
    joblib.dump(tm["control_model"], DATA_OUT_DIR / "aurix_uplift_model_control_xgb.joblib")
    np.save("aurix_uplift_scores_xgb.npy", uplift_scores)
    
    pd.DataFrame({"treatment": treatment, "y": y, "uplift": uplift_scores}).to_csv("aurix_uplift_eval_xgb.csv", index=False)

    print("Training complete (XGB).")
    print("Artifacts:")
    print(" - aurix_churn_model_xgb.joblib")
    print(" - aurix_churn_eval_xgb.json")
    print(" - xgb_roc.pdf, xgb_profit_curve.pdf, xgb_feature_importance.pdf")
    print(" - qini_placeholder.pdf, aurix_uplift_scores_xgb.npy, aurix_uplift_eval_xgb.csv")
    
    print(" - aurix_uplift_model_treated_xgb.joblib, aurix_uplift_model_control_xgb.joblib")

# Main call
if __name__ == '__main__':
    main()