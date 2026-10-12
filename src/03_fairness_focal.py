"""
03 - Fairness disparities of the focal model and of the deployed COMPAS
score (Section 4.2, Table 3, Figure 1).

Part A  Table 3: DPD, EOD, group-wise FPR/TPR and their gaps, AUC, DIR,
        for XGBoost (full features) and COMPAS (decile >= 7) on the
        identical seed-42 test split.
Part B  Figure 1: FPR by decision threshold and the group-specific
        threshold that equalizes FPRs.
Part C  Matched marking rates: FPR gaps when both systems mark the same
        share of defendants.
Part D  Group-wise calibration (ECE) over 30 seeds, paired gap AA - CA,
        for the focal model and for COMPAS decile/10.

Outputs: table3_fairness.csv, figure1_threshold_fpr.png,
         matched_marking_rates.csv, ece_gap_seeds.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import false_positive_rate

import config as C
from common import (by_group, disparate_impact_ratio, dpd, ece_gap, encode, eod,
                    fmt, fpr_gap, load_compas, make_model, out, paired_summary,
                    save, split, tpr_by_group)


def summary(label, y, yhat, prob, s):
    f = by_group(y, yhat, s, false_positive_rate)
    t = tpr_by_group(y, yhat, s)
    return {"System": label, "AUC": roc_auc_score(y, prob), "DPD": dpd(y, yhat, s),
            "EOD": eod(y, yhat, s), "FPR AA": f[C.DISADVANTAGED], "FPR CA": f["Caucasian"],
            "FPR gap": f[C.DISADVANTAGED] - f["Caucasian"],
            "TPR AA": t[C.DISADVANTAGED], "TPR CA": t["Caucasian"],
            "DIR": disparate_impact_ratio(yhat, s), "Marking rate": np.mean(yhat)}


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    X = encode(df, C.FEATURES_FULL)
    idx_tr, idx_te = split(df, C.SEED_DESCRIPTIVE)
    model = make_model("xgb", C.SEED_DESCRIPTIVE).fit(X.loc[idx_tr], y.loc[idx_tr])

    yte, ste = y.loc[idx_te].values, df.loc[idx_te, C.PROTECTED].values
    prob = model.predict_proba(X.loc[idx_te])[:, 1]
    yhat = (prob >= 0.5).astype(int)
    decile = df.loc[idx_te, "decile_score"].values
    y_compas = (decile >= C.COMPAS_HIGH_DECILE).astype(int)

    # --- Part A: Table 3 -----------------------------------------------------
    tab3 = pd.DataFrame([summary(f"COMPAS (decile >= {C.COMPAS_HIGH_DECILE})", yte, y_compas, decile / 10, ste),
                         summary("XGBoost (full features)", yte, yhat, prob, ste)]).round(3)
    print("Table 3\n" + tab3.to_string(index=False))
    save(tab3, "table3_fairness.csv")

    # --- Part B: threshold sweep and equalizing threshold (Figure 1) ---------
    rows = []
    for t in [0.30, 0.40, 0.50, 0.60]:
        yt = (prob >= t).astype(int)
        f = by_group(yte, yt, ste, false_positive_rate)
        rows.append({"threshold": t, "FPR_AA": f[C.DISADVANTAGED], "FPR_CA": f["Caucasian"],
                     "gap": f[C.DISADVANTAGED] - f["Caucasian"]})
    sweep = pd.DataFrame(rows).round(3)
    print("\nFPR by threshold\n" + sweep.to_string(index=False))

    def fpr_group(group, t):
        m = ste == group
        neg = yte[m] == 0
        return ((prob[m] >= t) & neg).sum() / neg.sum()

    fpr_aa_ref = fpr_group(C.DISADVANTAGED, 0.5)
    t_ca = brentq(lambda t: fpr_group("Caucasian", t) - fpr_aa_ref, 0.01, 0.99)
    print(f"Threshold for Caucasian defendants that equalizes FPR at the AA level ({fpr_aa_ref:.3f}): {t_ca:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sweep.threshold, sweep.FPR_AA, "o-", color="black", label="African American")
        ax.plot(sweep.threshold, sweep.FPR_CA, "s--", color="0.5", label="Caucasian")
        ax.axvline(t_ca, color="0.3", ls=":", label=f"equalizing threshold ({t_ca:.2f})")
        ax.set_xlabel("decision threshold")
        ax.set_ylabel("false positive rate")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out("figure1_threshold_fpr.png"), dpi=300)
        print("saved: outputs/figure1_threshold_fpr.png")
    except Exception as e:
        print("figure skipped:", e.__class__.__name__)

    # --- Part C: matched marking rates ---------------------------------------
    rate_c, rate_s = y_compas.mean(), yhat.mean()
    t_match = np.quantile(prob, 1 - rate_c)
    y_s_matched = (prob >= t_match).astype(int)
    best_decile = min(range(1, 11), key=lambda c: abs((decile >= c).mean() - rate_s))
    y_c_matched = (decile >= best_decile).astype(int)
    matched = pd.DataFrame([
        {"comparison": "default cut-offs", "rate COMPAS": rate_c, "rate surrogate": rate_s,
         "FPR gap COMPAS": fpr_gap(yte, y_compas, ste), "FPR gap surrogate": fpr_gap(yte, yhat, ste)},
        {"comparison": "surrogate matched to COMPAS rate", "rate COMPAS": rate_c, "rate surrogate": y_s_matched.mean(),
         "FPR gap COMPAS": fpr_gap(yte, y_compas, ste), "FPR gap surrogate": fpr_gap(yte, y_s_matched, ste)},
        {"comparison": f"COMPAS at decile >= {best_decile}", "rate COMPAS": y_c_matched.mean(), "rate surrogate": rate_s,
         "FPR gap COMPAS": fpr_gap(yte, y_c_matched, ste), "FPR gap surrogate": fpr_gap(yte, yhat, ste)},
    ]).round(3)
    print("\nMatched marking rates\n" + matched.to_string(index=False))
    save(matched, "matched_marking_rates.csv")

    # --- Part D: calibration over 30 seeds -----------------------------------
    gaps = []
    for seed in range(C.N_SEEDS):
        itr, ite = split(df, seed)
        m = make_model("xgb", seed).fit(X.loc[itr], y.loc[itr])
        p = m.predict_proba(X.loc[ite])[:, 1]
        s = df.loc[ite, C.PROTECTED].values
        gaps.append({"seed": seed,
                     "ece_gap_xgb": ece_gap(y.loc[ite], p, s),
                     "ece_gap_compas": ece_gap(y.loc[ite], df.loc[ite, "decile_score"].values / 10, s)})
    gaps = pd.DataFrame(gaps)
    print("\nPaired ECE gap (AA - CA) over 30 seeds")
    print("  XGBoost focal :", fmt(paired_summary(gaps.ece_gap_xgb)))
    print("  COMPAS        :", fmt(paired_summary(gaps.ece_gap_compas)))
    save(gaps, "ece_gap_seeds.csv")


if __name__ == "__main__":
    main()
