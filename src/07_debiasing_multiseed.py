"""
07 - The de-biasing experiment over 30 seeds (Sections 3.5, 4.4;
Table 5, Figure 3).

For each seed (0..29), each feature set (medium, full), each alpha in
{0, 0.05, ..., 1} and each model class (XGBoost, random forest, logistic
regression) the mixed proxy p(alpha) replaces prior offence count. The
residualization is fitted on the training partition only. Two control
transformations are run alongside:
    resid        p(alpha) as described in Section 3.5
    resid_round  p(alpha) rounded back to the integer grid
Recorded per fit: DPD, FPR gap, accuracy, ECE gap. Recorded per alpha:
attribute recoverability from p(alpha) alone (depth-6 tree, unrestricted
tree, logistic regression).

Aggregation (unit of analysis = seed):
    Table 5      XGBoost, medium set, resid: mean over alpha > 0 minus alpha = 0,
                 bootstrap CI, two-sided Wilcoxon
    controls     the same contrast for every model x transform x feature set
    dose-response per-seed Spearman rho(alpha, DPD) and location of the
                 largest DPD jump
    recoverability baseline vs transformed, three probes
    age          the same residualization applied to the second proxy
                 (full set, alpha in {0, 1} and mean over alpha > 0)

Runtime: roughly 30-60 minutes.
Outputs: debiasing_raw.csv, recoverability_raw.csv, table5.csv,
         debiasing_controls.csv, dose_response.csv, recoverability_summary.csv,
         age_residualization.csv, figure3_frontier.png
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score

import config as C
from common import (dpd, ece_gap, encode, fmt, fpr_gap, group_indicator, load_compas,
                    make_model, mix, out, paired_summary, recoverability, residualize,
                    save, split)

MODELS = ["xgb", "rf", "lr"]
SETS = {"medium": C.FEATURES_MEDIUM, "full": C.FEATURES_FULL}


def run_frontier(df):
    y = df[C.TARGET].astype(int)
    rows, rec_rows = [], []
    for seed in range(C.N_SEEDS):
        itr, ite = split(df, seed)
        tr, te = df.loc[itr], df.loc[ite]
        ytr, yte, ste = y.loc[itr], y.loc[ite], te[C.PROTECTED].values
        gtr, gte = group_indicator(tr).values, group_indicator(te).values
        r_tr, r_te, delta = residualize(tr, te, C.PROXY)
        for alpha in C.ALPHAS:
            p_tr, p_te = mix(tr[C.PROXY], r_tr, alpha), mix(te[C.PROXY], r_te, alpha)
            for tf, (ptr, pte) in {"resid": (p_tr, p_te), "resid_round": (p_tr.round(), p_te.round())}.items():
                rec_rows.append({"seed": seed, "alpha": alpha, "transform": tf, "delta_hat": delta,
                                 "rec_depth6": recoverability(ptr, gtr, pte, gte, C.RECOV_DEPTH, seed),
                                 "rec_unrestricted": recoverability(ptr, gtr, pte, gte, None, seed),
                                 "rec_lr": recoverability(ptr, gtr, pte, gte, learner="lr")})
                for set_name, cols in SETS.items():
                    Xtr, Xte = encode(tr, cols), encode(te, cols)
                    Xtr[C.PROXY], Xte[C.PROXY] = ptr.values, pte.values
                    for m in MODELS:
                        model = make_model(m, seed).fit(Xtr, ytr)
                        yhat = model.predict(Xte)
                        prob = model.predict_proba(Xte)[:, 1]
                        rows.append({"seed": seed, "alpha": alpha, "transform": tf, "set": set_name, "model": m,
                                     "DPD": dpd(yte, yhat, ste), "FPR_gap": fpr_gap(yte, yhat, ste),
                                     "accuracy": accuracy_score(yte, yhat), "ECE_gap": ece_gap(yte, prob, ste)})
        print(f"seed {seed} done")
    return pd.DataFrame(rows), pd.DataFrame(rec_rows)


def paired(sub, col):
    """Per seed: mean over alpha > 0 minus alpha = 0."""
    return np.array([s.loc[s.alpha > 0, col].mean() - s.loc[s.alpha == 0, col].iloc[0]
                     for _, s in sub.groupby("seed")])


def run_age(df):
    """Residualization of the second proxy (age) on the full set."""
    y = df[C.TARGET].astype(int)
    rows = []
    for seed in range(C.N_SEEDS):
        itr, ite = split(df, seed)
        tr, te = df.loc[itr], df.loc[ite]
        gtr, gte = group_indicator(tr).values, group_indicator(te).values
        r_tr, r_te, delta = residualize(tr, te, C.SECOND_PROXY)
        for alpha in C.ALPHAS:
            a_tr, a_te = mix(tr[C.SECOND_PROXY], r_tr, alpha), mix(te[C.SECOND_PROXY], r_te, alpha)
            rec6 = recoverability(a_tr, gtr, a_te, gte, C.RECOV_DEPTH, seed)
            recu = recoverability(a_tr, gtr, a_te, gte, None, seed)
            Xtr, Xte = encode(tr, C.FEATURES_FULL), encode(te, C.FEATURES_FULL)
            Xtr[C.SECOND_PROXY], Xte[C.SECOND_PROXY] = a_tr.values, a_te.values
            for m in MODELS:
                yhat = make_model(m, seed).fit(Xtr, y.loc[itr]).predict(Xte)
                rows.append({"seed": seed, "alpha": alpha, "model": m, "delta_hat": delta,
                             "DPD": dpd(y.loc[ite], yhat, te[C.PROTECTED].values),
                             "rec_depth6": rec6, "rec_unrestricted": recu})
        print(f"age seed {seed} done")
    return pd.DataFrame(rows)


def main():
    df = load_compas()
    if out("debiasing_raw.csv").exists():
        raw, rec = pd.read_csv(out("debiasing_raw.csv")), pd.read_csv(out("recoverability_raw.csv"))
        print("using existing outputs/debiasing_raw.csv (delete to recompute)")
    else:
        raw, rec = run_frontier(df)
        save(raw, "debiasing_raw.csv")
        save(rec, "recoverability_raw.csv")

    # --- Table 5 -----------------------------------------------------------------
    focal = raw.query("model == 'xgb' and transform == 'resid' and set == 'medium'")
    print("\nTable 5 (XGBoost, medium set, mean over alpha > 0 minus alpha = 0)")
    t5 = []
    for col in ["DPD", "FPR_gap", "accuracy"]:
        d = paired_summary(paired(focal, col))
        d["outcome"] = col
        t5.append(d)
        print(f"  {col:<10}", fmt(d))
    base = focal[focal.alpha == 0].DPD.mean()
    print(f"  baseline DPD {base:.3f}; relative effect {100 * t5[0]['delta'] / base:.1f}%")
    save(pd.DataFrame(t5), "table5.csv")

    # --- Controls: every model x transform x set --------------------------------
    ctrl = []
    for (set_name, tf, m), sub in raw.groupby(["set", "transform", "model"]):
        for col in ["DPD", "FPR_gap", "accuracy", "ECE_gap"]:
            d = paired_summary(paired(sub, col))
            d.update(set=set_name, transform=tf, model=m, outcome=col)
            ctrl.append(d)
    ctrl = pd.DataFrame(ctrl)[["set", "transform", "model", "outcome", "delta", "ci_low", "ci_high", "p", "n_pos", "n_neg"]]
    print("\nControls\n" + ctrl.round(4).to_string(index=False))
    save(ctrl, "debiasing_controls.csv")

    # --- Dose-response (focal specification) -------------------------------------
    rhos, jumps = [], []
    for _, s in focal.groupby("seed"):
        s = s.sort_values("alpha").reset_index(drop=True)
        g = s[s.alpha >= 0.05]
        rho = stats.spearmanr(g.alpha, g.DPD).correlation
        rhos.append(0.0 if np.isnan(rho) else rho)
        d = s.DPD.diff().abs().values[1:]
        jumps.append(s.alpha.values[1 + int(np.argmax(d))] if d.max() > 0 else np.nan)
    rhos = np.array(rhos)
    print(f"\nDose-response: Spearman rho(alpha, DPD) over alpha >= 0.05: mean {rhos.mean():.3f}, "
          f"positive in {int((rhos > 0).sum())}/{len(rhos)} seeds")
    print("location of the largest DPD jump per seed:", pd.Series(jumps).value_counts().sort_index().to_dict())
    print("mean DPD by alpha:\n" + focal.groupby("alpha").DPD.mean().round(4).to_string())
    save(pd.DataFrame({"seed": range(len(rhos)), "spearman_rho": rhos, "jump_alpha": jumps}), "dose_response.csv")

    # --- Recoverability -----------------------------------------------------------
    rs = []
    for tf, g in rec.groupby("transform"):
        for probe in ["rec_depth6", "rec_unrestricted", "rec_lr"]:
            a0 = g[g.alpha == 0][probe]
            a1 = g[g.alpha == 1.0][probe]
            amean = g[g.alpha > 0].groupby("seed")[probe].mean()
            rs.append({"transform": tf, "probe": probe, "alpha0": a0.mean(), "alpha1": a1.mean(),
                       "alpha1_min": a1.min(), "mean_alpha_gt0": amean.mean(),
                       "n_pos": int(((a1.values - a0.values) > 0).sum())})
    rs = pd.DataFrame(rs)
    print("\nRecoverability of group membership from p(alpha) alone\n" + rs.round(3).to_string(index=False))
    save(rs, "recoverability_summary.csv")

    # --- Age residualization ---------------------------------------------------------
    if out("age_residualization_raw.csv").exists():
        age = pd.read_csv(out("age_residualization_raw.csv"))
    else:
        age = run_age(df)
        save(age, "age_residualization_raw.csv")
    print(f"\nAge residualization (full set), delta_hat = {age.delta_hat.mean():.3f}")
    rows = []
    for m, sub in age.groupby("model"):
        d_mean = paired_summary(paired(sub, "DPD"))
        d_mean.update(model=m, contrast="mean alpha>0 - alpha=0")
        rows.append(d_mean)
        at_one = [s.loc[s.alpha == 1.0, "DPD"].iloc[0] - s.loc[s.alpha == 0, "DPD"].iloc[0]
                  for _, s in sub.groupby("seed")]
        d1 = paired_summary(np.array(at_one))
        d1.update(model=m, contrast="alpha=1 - alpha=0")
        rows.append(d1)
        print(f"  {m:<4} mean>0: {fmt(d_mean)}\n       alpha=1: {fmt(d1)}")
    r0 = age[(age.model == "xgb") & (age.alpha == 0)]
    r1 = age[(age.model == "xgb") & (age.alpha == 1.0)]
    print(f"  recoverability depth-6 {r0.rec_depth6.mean():.3f} -> {r1.rec_depth6.mean():.3f}; "
          f"unrestricted {r1.rec_unrestricted.mean():.3f}")
    save(pd.DataFrame(rows), "age_residualization.csv")

    # --- Figure 3 ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = focal.groupby("alpha")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        panels = [("DPD", "demographic parity difference", "(a) Demographic parity difference"),
                  ("FPR_gap", "FPR gap", "(b) False positive rate gap")]
        for ax, (col, lab, title) in zip(axes, panels):
            m, s = g[col].mean(), g[col].std() / np.sqrt(C.N_SEEDS) * 1.96
            ax.fill_between(m.index, m - s, m + s, color="0.85", lw=0)
            ax.plot(m.index, m, "o-", color="black", ms=3)
            ax.axvline(0.05, color="0.4", ls=":")
            ax.axhline(C.DISPARITY_THRESHOLD, color="0.4", ls="--")
            ax.set_xlabel("de-biasing intensity α")
            ax.set_ylabel(lab)
            ax.set_title(title, fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(out("figure3_frontier.png"), dpi=300)
        print("saved: outputs/figure3_frontier.png")
    except Exception as e:
        print("figure skipped:", e.__class__.__name__)


if __name__ == "__main__":
    main()
