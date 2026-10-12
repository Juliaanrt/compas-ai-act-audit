"""
09 - Temporal robustness (Section 4.5, Table 6).

Six follow-up cutoff dates, each median-split into an early (W1) and a
late (W2) window. Screenings from April 2014 on are excluded throughout
(censoring artifact, Section 3.1).

Part A  SHAP ranking of the medium-set XGBoost model in each of the twelve
        windows and between-window Spearman correlation of the ranks.
Part B  Rank stability under 500 bootstrap resamples of the sample
        restricted to screenings before April 2014.
Part C  Table 6: window sensitivity of the de-biasing step effect
        (mean DPD over alpha > 0 minus alpha = 0, 20 seeds per window,
        residualization fitted on the training partition, one-sided
        Wilcoxon in the direction established in Table 5).

Runtime: Part C roughly 20 minutes.
Outputs: shap_windows.csv, shap_bootstrap.csv, table6_window_sensitivity.csv
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

import config as C
from common import (dpd, encode, load_compas, make_model, mean_abs_rank, mix, residualize,
                    save, shap_values_xgb, split)


def shap_ranks(d, seed):
    X, y = encode(d, C.FEATURES_MEDIUM), d[C.TARGET].astype(int)
    itr, ite = split(d, seed)
    m = make_model("xgb", seed).fit(X.loc[itr], y.loc[itr])
    return mean_abs_rank(shap_values_xgb(m, X.loc[ite]), X.columns)["rank"]


def windows(dft):
    for cutoff in C.WINDOW_CUTOFFS:
        dc = dft[dft.screening_date < pd.Timestamp(cutoff)]
        med = dc.screening_date.median()
        yield cutoff, "W1", dc[dc.screening_date <= med]
        yield cutoff, "W2", dc[dc.screening_date > med]


def step_effect(sub):
    y = sub[C.TARGET].astype(int)
    deltas = []
    for seed in range(C.N_SEEDS_WINDOW):
        itr, ite = split(sub, seed)
        tr, te = sub.loc[itr], sub.loc[ite]
        r_tr, r_te, _ = residualize(tr, te, C.PROXY)
        d = {}
        for a in C.ALPHAS:
            Xtr, Xte = encode(tr, C.FEATURES_MEDIUM), encode(te, C.FEATURES_MEDIUM)
            Xtr[C.PROXY], Xte[C.PROXY] = mix(tr[C.PROXY], r_tr, a).values, mix(te[C.PROXY], r_te, a).values
            yhat = make_model("xgb", seed).fit(Xtr, y.loc[itr]).predict(Xte)
            d[a] = dpd(y.loc[ite], yhat, te[C.PROTECTED].values)
        deltas.append(np.mean([v for a, v in d.items() if a > 0]) - d[0.0])
    v = np.array(deltas)
    return v.mean(), int((v > 0).sum()), stats.wilcoxon(v, alternative="greater").pvalue


def main():
    df = load_compas()
    dft = df.copy()
    dft["screening_date"] = pd.to_datetime(dft["compas_screening_date"], errors="coerce")
    dft = dft.dropna(subset=["screening_date"])
    dft = dft[dft.screening_date < pd.Timestamp(C.CENSORING_START)]

    # --- Part A: SHAP ranks per window -------------------------------------------
    R = pd.DataFrame({f"{c} {w}": shap_ranks(sub, C.SEED_DESCRIPTIVE) for c, w, sub in windows(dft)}).T
    print("SHAP ranks per window\n" + R.astype(int).to_string())
    n_ok = int(((R.priors_count == 1) & (R.age == 2)).sum())
    keys = list(R.index)
    rhos = [stats.spearmanr(R.loc[a], R.loc[b]).correlation for i, a in enumerate(keys) for b in keys[i + 1:]]
    print(f"prior offence count rank 1 and age rank 2 in {n_ok}/{len(R)} windows; "
          f"{{priors, age}} are the two leading features in {int(((R.priors_count <= 2) & (R.age <= 2)).sum())}/{len(R)}; "
          f"between-window Spearman rho {np.nanmin(rhos):.2f}-{np.nanmax(rhos):.2f}")
    save(R.reset_index().rename(columns={"index": "window"}), "shap_windows.csv")

    # --- Part B: bootstrap ---------------------------------------------------------
    rng = np.random.default_rng(C.SEED_DESCRIPTIVE)
    base = dft.reset_index(drop=True)
    rows = []
    for i in range(C.N_BOOT):
        b = base.iloc[rng.choice(len(base), len(base), replace=True)].reset_index(drop=True)
        r = shap_ranks(b, i)
        rows.append({"resample": i, **r.to_dict()})
    boot = pd.DataFrame(rows)
    exact = int(((boot.priors_count == 1) & (boot.age == 2)).sum())
    top2 = int(((boot.priors_count <= 2) & (boot.age <= 2)).sum())
    print(f"\nBootstrap (n = {len(base)}, {C.N_BOOT} resamples): priors first and age second in {exact}; "
          f"{{priors, age}} leading in {top2}")
    save(boot, "shap_bootstrap.csv")

    # --- Part C: Table 6 ---------------------------------------------------------------
    rows = []
    for cutoff, w, sub in windows(dft):
        m, npos, p = step_effect(sub)
        rows.append({"cutoff": cutoff, "window": w, "n": len(sub), "dDPD": round(m, 4),
                     "positive_seeds": f"{npos}/{C.N_SEEDS_WINDOW}", "p_wilcoxon_greater": round(p, 3)})
        print(rows[-1])
    t6 = pd.DataFrame(rows)
    print("\nTable 6\n" + t6.to_string(index=False))
    save(t6, "table6_window_sensitivity.csv")


if __name__ == "__main__":
    main()
