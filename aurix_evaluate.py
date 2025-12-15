# # aurix_evaluate.py
# # Aggregates model and uplift results, computes profit/Qini summaries using shared helpers,
# # and exports LaTeX-ready tables plus two plot styles (academic & industry).

# import json
# import joblib
# import sys, os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from dataclasses import dataclass
# from matplotlib.ticker import PercentFormatter

# # Ensure src/ is on path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "data"
# FIG_DIR = DATA_DIR / "figs"
# DATA_OUT_DIR = DATA_DIR / "data_out"

# FIG_DIR.mkdir(parents=True, exist_ok=True)
# DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# # Use single source of truth for profit math
# from src.api.profit_loss import (
#     ProfitConfig,
#     profit_curve_from_scores,
#     argmax_profit,
# )

# def load_cfg() -> ProfitConfig:
#     return ProfitConfig(
#         margin_per_customer=float(os.getenv("AURIX_MARGIN", 100.0)),
#         contact_cost=float(os.getenv("AURIX_CONTACT_COST", 5.0)),
#         max_contact_rate=float(os.getenv("AURIX_MAX_CONTACT_RATE", 0.30)),
#         uplift_effect_guess=float(os.getenv("AURIX_UPLIFT_GUESS", 0.20)),
#     )

# def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
#     return float(np.trapz(y, x))

# def latex_table(df: pd.DataFrame, caption: str, label: str, out_path: str) -> None:
#     with open(out_path, "w") as f:
#         f.write(df.to_latex(index=False, escape=True, caption=caption, label=label, float_format="%.4f"))

# def qini_from_components(uplift_scores: np.ndarray, treatment: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
#     order = np.argsort(-uplift_scores)
#     t_sorted = treatment[order]
#     y_sorted = y_true[order]
#     n = len(y_sorted)
#     ks = np.arange(1, n + 1)
#     inc = np.cumsum(2 * t_sorted * y_sorted - t_sorted - y_sorted)
#     return pd.DataFrame({"k": ks, "rate": ks / n, "qini": inc})

# def plot_profit(curve: pd.DataFrame, out_path: str, style: str = "academic"):
#     plt.figure(figsize=(6, 4))
#     if style == "industry":
#         plt.plot(curve["contact_rate"], curve["profit"], linewidth=2.5)
#     else:
#         plt.plot(curve["contact_rate"], curve["profit"])
#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
#     plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
#     plt.xlabel("Contact Rate")
#     plt.ylabel("Expected Profit")
#     plt.title("Profit Curve")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def plot_qini(curve: pd.DataFrame, out_path: str, style: str = "academic"):
#     plt.figure(figsize=(6, 4))
#     if style == "industry":
#         plt.plot(curve["rate"], curve["qini"], linewidth=2.5)
#     else:
#         plt.plot(curve["rate"], curve["qini"])
#     plt.axhline(0, linestyle="--")
#     plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
#     plt.xlabel("Targeted Rate")
#     plt.ylabel("Incremental Response (arb. units)")
#     plt.title("Qini Curve")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def main():
#     cfg = load_cfg()

#     features_path = DATA_OUT_DIR / "aurix_features.csv"
#     if not features_path.exists():
#         raise FileNotFoundError(
#             f"{features_path} not found. Run feature engineering first."
#         )

#     df = pd.read_csv(features_path)
#     if "churn_flag" not in df.columns:
#         raise ValueError("Expected 'churn_flag' in aurix_features.csv.")
#     y = df["churn_flag"].astype(int)
#     X = df.drop(columns=["churn_flag"], errors="ignore")

#     summaries = []

#     # Compute and export profit curves for any available model
#     for model_name, model_path, curve_out in [
#         ("baseline", "aurix_churn_model.joblib", "profit_curve_baseline.csv"),
#         ("xgb", "aurix_churn_model_xgb.joblib", "profit_curve_xgb.csv"),
#     ]:
#         if os.path.exists(model_path):
#             mdl = joblib.load(model_path)
#             scores = mdl.predict_proba(X)[:, 1]
#             curve = profit_curve_from_scores(scores, cfg)
#             curve.to_csv(curve_out, index=False)
#             k, rate, pmax = argmax_profit(curve)
#             summaries.append({"model": model_name, "max_profit": pmax, "contact_rate_at_max": rate})

#     profit_summary = pd.DataFrame(summaries)
#     if not profit_summary.empty:
#         profit_summary.to_csv("profit_summary.csv", index=False)
#         latex_table(
#             profit_summary.round({"max_profit": 2, "contact_rate_at_max": 3}),
#             caption="Profit summary across models.",
#             label="tab:profit_summary",
#             out_path="tables_profit.tex",
#         )

#     # Metrics jsons if available
#     metrics_all = {}
#     if os.path.exists("aurix_churn_eval.json"):
#         with open("aurix_churn_eval.json") as f:
#             data = json.load(f)
#             if isinstance(data, dict) and "auc" in next(iter(data.values()), {}):
#                 best = max(data, key=lambda k: data[k]["auc"])
#                 metrics_all["baseline"] = data[best]
#             else:
#                 metrics_all["baseline"] = data
#     if os.path.exists("aurix_churn_eval_xgb.json"):
#         with open("aurix_churn_eval_xgb.json") as f:
#             metrics_all["xgb"] = json.load(f)

#     # Uplift AUC (if artifacts exist)
#     uplift_rows = []
#     if os.path.exists("aurix_uplift_eval.csv"):
#         u = pd.read_csv("aurix_uplift_eval.csv")
#         q = qini_from_components(u["uplift"].to_numpy(), u["treatment"].to_numpy(), u["y"].to_numpy())
#         q.to_csv("qini_curve_baseline.csv", index=False)
#         uplift_rows.append({"model": "baseline", "auc_qini": auc_trapz(q["rate"].to_numpy(), q["qini"].to_numpy())})
#     if os.path.exists("aurix_uplift_eval_xgb.csv"):
#         u = pd.read_csv("aurix_uplift_eval_xgb.csv")
#         q = qini_from_components(u["uplift"].to_numpy(), u["treatment"].to_numpy(), u["y"].to_numpy())
#         q.to_csv("qini_curve_xgb.csv", index=False)
#         uplift_rows.append({"model": "xgb", "auc_qini": auc_trapz(q["rate"].to_numpy(), q["qini"].to_numpy())})

#     uplift_summary = pd.DataFrame(uplift_rows)
#     if not uplift_summary.empty:
#         uplift_summary.to_csv("uplift_summary.csv", index=False)
#         latex_table(
#             uplift_summary.round({"auc_qini": 3}),
#             caption="Qini AUC across models.",
#             label="tab:uplift_summary",
#             out_path="tables_uplift.tex",
#         )

#     # Plots (academic + industry)
#     if os.path.exists("profit_curve_baseline.csv"):
#         cb = pd.read_csv("profit_curve_baseline.csv")
#         plot_profit(cb, "profit_curve_academic_baseline.pdf", "academic")
#         plot_profit(cb, "profit_curve_industry_baseline.pdf", "industry")
#     if os.path.exists("profit_curve_xgb.csv"):
#         cx = pd.read_csv("profit_curve_xgb.csv")
#         plot_profit(cx, "profit_curve_academic_xgb.pdf", "academic")
#         plot_profit(cx, "profit_curve_industry_xgb.pdf", "industry")

#     if os.path.exists("qini_curve_baseline.csv"):
#         qb = pd.read_csv("qini_curve_baseline.csv")
#         plot_qini(qb, "qini_academic_baseline.pdf", "academic")
#         plot_qini(qb, "qini_industry_baseline.pdf", "industry")
#     if os.path.exists("qini_curve_xgb.csv"):
#         qx = pd.read_csv("qini_curve_xgb.csv")
#         plot_qini(qx, "qini_academic_xgb.pdf", "academic")
#         plot_qini(qx, "qini_industry_xgb.pdf", "industry")

#     print("Evaluation complete.")

# if __name__ == "__main__":
#     main()


###


# Aggregates model and uplift results, computes profit/Qini summaries using shared helpers,
# and exports LaTeX-ready tables plus two plot styles (academic & industry).

import json
import joblib
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from matplotlib.ticker import PercentFormatter
from src.models.profit_loss import (
    ProfitConfig,
    profit_curve_from_scores,
    argmax_profit,
    roi_at_k
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

def load_cfg() -> ProfitConfig:
    return ProfitConfig(
        margin_per_customer=float(os.getenv("AURIX_MARGIN", 100.0)),
        contact_cost=float(os.getenv("AURIX_CONTACT_COST", 5.0)),
        max_contact_rate=float(os.getenv("AURIX_MAX_CONTACT_RATE", 0.30)),
        uplift_effect_guess=float(os.getenv("AURIX_UPLIFT_GUESS", 0.20)),
    )

def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))

def latex_table(df: pd.DataFrame, caption: str, label: str, out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write(
            df.to_latex(
                index=False,
                escape=True,
                caption=caption,
                label=label,
                float_format="%.4f",
            )
        )

def qini_from_components(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """
    Generic Qini curve computation.
    uplift_scores: scores used for ranking (uplift or propensity).
    treatment: binary {0,1} indicator of treatment.
    y_true: binary {0,1} outcome.
    """
    order = np.argsort(-uplift_scores)
    t_sorted = treatment[order]
    y_sorted = y_true[order]
    n = len(y_sorted)
    ks = np.arange(1, n + 1)
    inc = np.cumsum(2 * t_sorted * y_sorted - t_sorted - y_sorted)
    return pd.DataFrame({"k": ks, "rate": ks / n, "qini": inc})

def plot_profit(curve: pd.DataFrame, out_path: Path, style: str = "academic"):
    plt.figure(figsize=(6, 4))
    if style == "industry":
        plt.plot(curve["contact_rate"], curve["profit"], linewidth=2.5)
    else:
        plt.plot(curve["contact_rate"], curve["profit"])
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Contact Rate")
    plt.ylabel("Expected Profit")
    # plt.title("Profit Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_qini(curve: pd.DataFrame, out_path: Path, style: str = "academic"):
    plt.figure(figsize=(6, 4))
    if style == "industry":
        plt.plot(curve["rate"], curve["qini"], linewidth=2.5)
    else:
        plt.plot(curve["rate"], curve["qini"])
    plt.axhline(0, linestyle="--")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Targeted Rate")
    plt.ylabel("Incremental Response (arb. units)")
    # plt.title("Qini Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    cfg = load_cfg()

    # Features live in data/data_out
    features_path = DATA_OUT_DIR / "aurix_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found. Run feature engineering first."
        )

    df = pd.read_csv(features_path)
    if "churn_flag" not in df.columns:
        raise ValueError("Expected 'churn_flag' in aurix_features.csv.")
    y = df["churn_flag"].astype(int)
    X = df.drop(columns=["churn_flag"], errors="ignore")

    summaries = []
    model_scores = {}

    # Compute and export profit curves for any available model
    for model_name, model_file, curve_file in [
        ("baseline", "aurix_churn_model.joblib", "profit_curve_baseline.csv"),
        ("xgb", "aurix_churn_model_xgb.joblib", "profit_curve_xgb.csv"),
    ]:
        model_path = DATA_OUT_DIR / model_file
        curve_out_path = DATA_OUT_DIR / curve_file

        if model_path.exists():
            mdl = joblib.load(model_path)
            scores = mdl.predict_proba(X)[:, 1]
            model_scores[model_name] = scores

            curve = profit_curve_from_scores(scores, cfg)
            curve.to_csv(curve_out_path, index=False)
            k, rate, pmax = argmax_profit(curve)
            summaries.append(
                {
                    "model": model_name,
                    "max_profit": pmax,
                    "contact_rate_at_max": rate,
                }
            )

    profit_summary = pd.DataFrame(summaries)
    if not profit_summary.empty:
        profit_summary_path = DATA_OUT_DIR / "profit_summary.csv"
        profit_summary.to_csv(profit_summary_path, index=False)
        latex_table(
            profit_summary.round({"max_profit": 2, "contact_rate_at_max": 3}),
            caption="Profit summary across models.",
            label="tab:profit_summary",
            out_path=str(DATA_OUT_DIR / "tables_profit.tex"),
        )

    # Metrics jsons if available
    metrics_all = {}
    baseline_eval_path = DATA_OUT_DIR / "aurix_churn_eval.json"
    xgb_eval_path = DATA_OUT_DIR / "aurix_churn_eval_xgb.json"

    if baseline_eval_path.exists():
        with open(baseline_eval_path) as f:
            data = json.load(f)
            if isinstance(data, dict) and "auc" in next(iter(data.values()), {}):
                best = max(data, key=lambda k: data[k]["auc"])
                metrics_all["baseline"] = data[best]
            else:
                metrics_all["baseline"] = data
    if xgb_eval_path.exists():
        with open(xgb_eval_path) as f:
            metrics_all["xgb"] = json.load(f)


# ------------------------------------------------------------------
    # Uplift / Qini analysis (XGB two-model uplift)
    # ------------------------------------------------------------------
    uplift_rows = []

    # Baseline uplift (if you ever add one) could go here similarly.

    uplift_eval_xgb_path = DATA_OUT_DIR / "aurix_uplift_eval_xgb.csv"
    if uplift_eval_xgb_path.exists():
        u = pd.read_csv(uplift_eval_xgb_path)
        q = qini_from_components(
            uplift_scores=u["uplift"].to_numpy(),
            treatment=u["treatment"].to_numpy(),
            y_true=u["y"].to_numpy(),
        )

        # Save Qini curve points
        qini_curve_xgb_csv = DATA_OUT_DIR / "qini_curve_xgb.csv"
        q.to_csv(qini_curve_xgb_csv, index=False)
        print(f"[aurix] Saved Qini curve points to {qini_curve_xgb_csv}")

        # Aggregate Qini AUC for summary table
        auc_q = auc_trapz(q["rate"].to_numpy(), q["qini"].to_numpy())
        uplift_rows.append({"model": "xgb", "auc_qini": auc_q})

        # Plot Qini curve figure
        qini_curve_xgb_pdf = FIG_DIR / "qini_curve_xgb.pdf"
        plot_qini(q, qini_curve_xgb_pdf, style="academic")
        print(f"[aurix] Saved Qini curve plot to {qini_curve_xgb_pdf}")
    else:
        print(f"[aurix] No uplift eval file at {uplift_eval_xgb_path}; skipping Qini.")
    
    uplift_summary = pd.DataFrame(uplift_rows)
    if not uplift_summary.empty:
        uplift_summary_path = DATA_OUT_DIR / "uplift_summary.csv"
        uplift_summary.to_csv(uplift_summary_path, index=False)
        print(f"[aurix] Wrote uplift summary to {uplift_summary_path}")


    # # Uplift AUC (if artifacts exist)
    # uplift_rows = []

    # # Optional baseline uplift (if ever produced)
    # uplift_eval_baseline = DATA_OUT_DIR / "aurix_uplift_eval.csv"
    # if uplift_eval_baseline.exists():
    #     u = pd.read_csv(uplift_eval_baseline)
    #     q = qini_from_components(
    #         u["uplift"].to_numpy(),
    #         u["treatment"].to_numpy(),
    #         u["y"].to_numpy(),
    #     )
    #     qini_baseline_path = DATA_OUT_DIR / "qini_curve_baseline.csv"
    #     q.to_csv(qini_baseline_path, index=False)
    #     uplift_rows.append(
    #         {
    #             "model": "baseline",
    #             "auc_qini": auc_trapz(
    #                 q["rate"].to_numpy(), q["qini"].to_numpy()
    #             ),
    #         }
    #     )

    # # XGB uplift + propensity Qini
    # uplift_eval_xgb = DATA_OUT_DIR / "aurix_uplift_eval_xgb.csv"
    # qini_uplift_path = DATA_OUT_DIR / "qini_curve_xgb_uplift.csv"
    # qini_propensity_path = DATA_OUT_DIR / "qini_curve_xgb_propensity.csv"

    # if uplift_eval_xgb.exists():
    #     u = pd.read_csv(uplift_eval_xgb)

    #     uplift_scores = u["uplift"].to_numpy()
    #     treatment = u["treatment"].to_numpy().astype(int)
    #     y_uplift = u["y"].to_numpy().astype(int)

    #     # Uplift-based Qini
    #     q_uplift = qini_from_components(uplift_scores, treatment, y_uplift)
    #     q_uplift.to_csv(qini_uplift_path, index=False)
    #     uplift_rows.append(
    #         {
    #             "model": "xgb_uplift",
    #             "auc_qini": auc_trapz(
    #                 q_uplift["rate"].to_numpy(),
    #                 q_uplift["qini"].to_numpy(),
    #             ),
    #         }
    #     )

    #     # Propensity-based Qini using XGB churn scores
    #     churn_scores = model_scores.get("xgb")
    #     if churn_scores is not None:
    #         n = min(len(churn_scores), len(treatment), len(y_uplift))
    #         q_prop = qini_from_components(
    #             churn_scores[:n],
    #             treatment[:n],
    #             y_uplift[:n],
    #         )
    #         q_prop.to_csv(qini_propensity_path, index=False)
    #         uplift_rows.append(
    #             {
    #                 "model": "xgb_propensity",
    #                 "auc_qini": auc_trapz(
    #                     q_prop["rate"].to_numpy(),
    #                     q_prop["qini"].to_numpy(),
    #                 ),
    #             }
    #         )
    #     else:
    #         print(
    #             "Warning: XGB churn scores not available; "
    #             "skipping propensity-based Qini."
    #         )

    # uplift_summary = pd.DataFrame(uplift_rows)
    # if not uplift_summary.empty:
    #     uplift_summary_path = DATA_OUT_DIR / "uplift_summary.csv"
    #     uplift_summary.to_csv(uplift_summary_path, index=False)

    # Plots (academic + industry)
    # Profit curves
    profit_curve_baseline_path = DATA_OUT_DIR / "profit_curve_baseline.csv"
    profit_curve_xgb_path      = DATA_OUT_DIR / "profit_curve_xgb.csv"
    qini_baseline_path         = DATA_OUT_DIR / "qini_curve_baseline.csv"
    qini_uplift_path           = DATA_OUT_DIR / "qini_curve_xgb.csv"

    if profit_curve_baseline_path.exists():
        cb = pd.read_csv(profit_curve_baseline_path)
        plot_profit(cb, FIG_DIR / "profit_curve_academic_baseline.pdf", "academic")
        plot_profit(cb, FIG_DIR / "profit_curve_industry_baseline.pdf", "industry")

    if profit_curve_xgb_path.exists():
        cx = pd.read_csv(profit_curve_xgb_path)
        plot_profit(cx, FIG_DIR / "xgb_profit_curve.pdf", "academic")
        # if you also want an “industry” variant:
        # plot_profit(cx, FIG_DIR / "profit_curve_industry_xgb.pdf", "industry")

    # Qini curves
    if qini_baseline_path.exists():
        qb = pd.read_csv(qini_baseline_path)
        plot_qini(qb, FIG_DIR / "qini_academic_baseline.pdf", "academic")
        plot_qini(qb, FIG_DIR / "qini_industry_baseline.pdf", "industry")

    if qini_uplift_path.exists():
        qx = pd.read_csv(qini_uplift_path)
        plot_qini(qx, FIG_DIR / "qini_curve_xgb.pdf", "academic")
        # if you want a second styling:
        # plot_qini(qx, FIG_DIR / "qini_industry_xgb.pdf", "industry")

    # if qini_propensity_path.exists():
    #     qx_p = pd.read_csv(qini_propensity_path)
    #     plot_qini(qx_p, FIG_DIR / "qini_propensity_academic.pdf", "academic")
    #     plot_qini(qx_p, FIG_DIR / "qini_propensity_industry.pdf", "industry")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()