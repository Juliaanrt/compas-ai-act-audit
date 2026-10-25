"""
Shared helpers: data loading, encoding, splitting, residualization,
fairness metrics, calibration, recoverability, and paired statistics.

All scripts build on these functions so that the operations described in
Sections 3.1-3.5 of the manuscript are implemented exactly once.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, false_positive_rate,
                               true_positive_rate)

import config as C

warnings.filterwarnings("ignore")

try:                      # shap is used only for the summary / waterfall plots
    import shap           # (Figure 2, case waterfalls) and for random-forest SHAP values
    HAS_SHAP = True
except Exception:       
    HAS_SHAP = False


# ---------------------------------------------------------------------------
# Data (Section 3.1)
# ---------------------------------------------------------------------------
def load_compas() -> pd.DataFrame:
    """ProPublica two-year file with the standard filter, restricted to the
    two largest groups. The original row labels are kept so that the two
    case-level defendants (config.CASE_A / CASE_B) stay addressable."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if C.DF_MAIN_FILE.exists():
        return pd.read_csv(C.DF_MAIN_FILE, index_col=0)
    if not C.COMPAS_FILE.exists():
        pd.read_csv(C.COMPAS_URL).to_csv(C.COMPAS_FILE, index=False)
    df = pd.read_csv(C.COMPAS_FILE)
    df = df[(df["days_b_screening_arrest"] <= 30)      # screening within +-30 days
            & (df["days_b_screening_arrest"] >= -30)   # of the arrest
            & (df["is_recid"] != -1)                   # recidivism flag valid
            & (df["c_charge_degree"] != "O")]          # ordinary traffic offences out
    df = df[df[C.PROTECTED].isin(C.GROUPS)].copy()
    df.to_csv(C.DF_MAIN_FILE, index=True)
    return df


def group_indicator(df: pd.DataFrame) -> pd.Series:
    """1 for the disadvantaged group, 0 otherwise."""
    return (df[C.PROTECTED] == C.DISADVANTAGED).astype(int)


def encode(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Numeric design matrix. sex -> Male=1, c_charge_degree -> F(elony)=1.
    Race is never part of the design matrix (Section 3.2)."""
    X = df[list(cols)].copy()
    if "sex" in X:
        X["sex"] = (X["sex"] == "Male").astype(float)
    if "c_charge_degree" in X:
        X["c_charge_degree"] = (X["c_charge_degree"] == "F").astype(float)
    return X.astype(float)


def split(df: pd.DataFrame, seed: int):
    """80/20 split stratified on the target (Section 3.2)."""
    return train_test_split(df.index, test_size=C.TEST_SIZE, random_state=seed,
                            stratify=df[C.TARGET])


# ---------------------------------------------------------------------------
# Models (Section 3.2): library defaults, no tuning
# ---------------------------------------------------------------------------
def make_model(name: str, seed: int):
    if name == "xgb":
        return XGBClassifier(n_estimators=100, random_state=seed,
                             eval_metric="logloss", verbosity=0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "lr":
        return LogisticRegression(max_iter=2000)
    raise ValueError(name)


MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


# ---------------------------------------------------------------------------
# Residualization (Section 3.5)
# ---------------------------------------------------------------------------
def residualize(tr: pd.DataFrame, te: pd.DataFrame, feature: str):
    """Least-squares residualization of `feature` on the binary group
    indicator. The regression is fitted on the training partition only and
    applied unchanged to the test partition (no test-set information enters
    the transformation). Returns (resid_train, resid_test, delta_hat)."""
    g_tr = group_indicator(tr).astype(float).to_frame("g")
    reg = LinearRegression().fit(g_tr, tr[feature])
    out = []
    for part in (tr, te):
        g = group_indicator(part).astype(float).to_frame("g")
        out.append(part[feature] - reg.predict(g))
    return out[0], out[1], float(reg.coef_[0])


def mix(original: pd.Series, resid: pd.Series, alpha: float) -> pd.Series:
    """p(alpha) = (1 - alpha) * original + alpha * residualized."""
    return (1 - alpha) * original + alpha * resid


# ---------------------------------------------------------------------------
# Fairness metrics (Section 3.4)
# ---------------------------------------------------------------------------
def dpd(y, yhat, s):
    return demographic_parity_difference(y, yhat, sensitive_features=s)


def eod(y, yhat, s):
    return equalized_odds_difference(y, yhat, sensitive_features=s)


def by_group(y, yhat, s, metric):
    mf = MetricFrame(metrics={"m": metric}, y_true=y, y_pred=yhat, sensitive_features=s)
    return mf.by_group["m"]


def fpr_gap(y, yhat, s):
    g = by_group(y, yhat, s, false_positive_rate)
    return g[C.DISADVANTAGED] - g["Caucasian"]


def tpr_by_group(y, yhat, s):
    return by_group(y, yhat, s, true_positive_rate)


def disparate_impact_ratio(yhat, s):
    """Ratio of positive prediction rates, Caucasian / African-American
    (four-fifths rule, Section 3.4)."""
    yhat, s = np.asarray(yhat), np.asarray(s)
    rate_aa = yhat[s == C.DISADVANTAGED].mean()
    rate_ca = yhat[s == "Caucasian"].mean()
    return rate_ca / rate_aa if rate_aa > 0 else np.nan


def ece(y, prob, bins: int = C.ECE_BINS) -> float:
    """Expected calibration error with equal-width bins (Section 3.4)."""
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (prob >= lo) & ((prob < hi) if i < bins - 1 else (prob <= hi))
        if m.any():
            err += m.mean() * abs(y[m].mean() - prob[m].mean())
    return err


def ece_gap(y, prob, s) -> float:
    """ECE(African-American) - ECE(Caucasian)."""
    y, prob, s = np.asarray(y), np.asarray(prob), np.asarray(s)
    aa = s == C.DISADVANTAGED
    return ece(y[aa], prob[aa]) - ece(y[~aa], prob[~aa])


# ---------------------------------------------------------------------------
# Attribute recoverability (Section 3.5, criterion of Feldman et al.)
# ---------------------------------------------------------------------------
def recoverability(f_tr, g_tr, f_te, g_te, depth=C.RECOV_DEPTH, seed: int = 0,
                   learner: str = "tree") -> float:
    """AUC of a probe predicting group membership from the feature alone.
    The probe is fitted on the training partition and scored on the test
    partition. depth=None gives an unrestricted tree."""
    f_tr = np.asarray(f_tr, dtype=float).reshape(-1, 1)
    f_te = np.asarray(f_te, dtype=float).reshape(-1, 1)
    if learner == "lr":
        probe = LogisticRegression(max_iter=2000)
    else:
        probe = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    probe.fit(f_tr, g_tr)
    return roc_auc_score(g_te, probe.predict_proba(f_te)[:, 1])


# ---------------------------------------------------------------------------
# SHAP (Section 3.3)
# ---------------------------------------------------------------------------
def shap_values_xgb(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values from XGBoost's own implementation (last column
    of pred_contribs is the bias term and is dropped)."""
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def shap_values_lr(model: LogisticRegression, X: pd.DataFrame, X_background: pd.DataFrame):
    """LinearSHAP with an independent (interventional) background:
    phi_j = coef_j * (x_j - mean_background_j)."""
    return (X - X_background.mean()).values * model.coef_[0]


def mean_abs_rank(sv: np.ndarray, columns) -> pd.DataFrame:
    imp = pd.Series(np.abs(sv).mean(axis=0), index=columns)
    return pd.DataFrame({"mean_abs_shap": imp, "rank": imp.rank(ascending=False).astype(int)})


# ---------------------------------------------------------------------------
# Paired statistics over seeds (Section 3.5)
# ---------------------------------------------------------------------------
def paired_summary(v, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean paired difference, bootstrap 95% CI over seeds and a two-sided
    Wilcoxon signed-rank test."""
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    p = stats.wilcoxon(v, alternative="two-sided").pvalue if np.any(v != 0) else np.nan
    return dict(delta=v.mean(), ci_low=np.percentile(boots, 2.5),
                ci_high=np.percentile(boots, 97.5), p=p,
                n_pos=int((v > 0).sum()), n_neg=int((v < 0).sum()), n=len(v))


def fmt(d: dict) -> str:
    return (f"Δ={d['delta']:+.4f}  CI[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p={d['p']:.4f}  pos {d['n_pos']}/{d['n']}  neg {d['n_neg']}/{d['n']}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def out(name: str) -> Path:
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    return C.OUTPUTS / name


def save(df: pd.DataFrame, name: str, index: bool = False):
    df.to_csv(out(name), index=index)
    print(f"saved: outputs/{name}")
