"""
04 - SHAP audit: global proxy identification, cross-model and cross-seed
robustness, case-level explanations (Sections 3.3, 4.3; Figure 2).

Part A  Figure 2: global SHAP summary of the focal model (seed 42).
Part B  Cross-model rank agreement on the medium (four-feature) set:
        LinearSHAP for logistic regression, TreeSHAP for random forest
        and XGBoost, same split, library defaults.
Part C  Rank stability of the focal model across the 30 seeds.
Part D  Supplementary (not shown as a figure in the manuscript): SHAP
        waterfalls for Case A and Case B (Section 4.3); minimal
        single-feature changes that flip each prediction (exhaustive
        integer sweep); integer-verified counterfactuals with DiCE
        (optional dependency).
Part E  Global SHAP importance at alpha = 0 and alpha = 1 (Section 4.4:
        attribution is not redistributed by the residualization).

Outputs: figure2_shap_summary.png, shap_cross_model.csv,
         shap_rank_stability_seeds.csv, waterfall_case_a/b.png,
         minimal_flips.csv, counterfactuals_case_A/B.csv, shap_alpha0_vs_alpha1.csv
"""

import numpy as np
import pandas as pd

import config as C
from common import (HAS_SHAP, encode, load_compas, make_model, mean_abs_rank, mix,
                    out, residualize, save, shap_values_lr, shap_values_xgb, split)


def verify_case(df, case):
    row = df.loc[case["index"]]
    for k in ("race", "age", "priors_count", "decile_score"):
        assert row[k] == case[k], f"case {case['index']}: {k}={row[k]} != {case[k]}"


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    X = encode(df, C.FEATURES_FULL)
    idx_tr, idx_te = split(df, C.SEED_DESCRIPTIVE)
    focal = make_model("xgb", C.SEED_DESCRIPTIVE).fit(X.loc[idx_tr], y.loc[idx_tr])

    # --- Part A: global summary (Figure 2) -----------------------------------
    sv = shap_values_xgb(focal, X.loc[idx_te])
    print("Global mean |SHAP|, focal model (test split)\n" + mean_abs_rank(sv, X.columns).round(4).to_string())
    if HAS_SHAP:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(sv, X.loc[idx_te], show=False)
        plt.tight_layout()
        plt.savefig(out("figure2_shap_summary.png"), dpi=300)
        plt.close()
        print("saved: outputs/figure2_shap_summary.png")

    # --- Part B: cross-model rank agreement (medium set) ---------------------
    Xm = encode(df, C.FEATURES_MEDIUM)
    lr = make_model("lr", C.SEED_DESCRIPTIVE).fit(Xm.loc[idx_tr], y.loc[idx_tr])
    xg = make_model("xgb", C.SEED_DESCRIPTIVE).fit(Xm.loc[idx_tr], y.loc[idx_tr])
    tab = pd.DataFrame({"LR": np.abs(shap_values_lr(lr, Xm.loc[idx_te], Xm.loc[idx_tr])).mean(0),
                        "XGB": np.abs(shap_values_xgb(xg, Xm.loc[idx_te])).mean(0)}, index=Xm.columns)
    if HAS_SHAP:
        import shap
        rf = make_model("rf", C.SEED_DESCRIPTIVE).fit(Xm.loc[idx_tr], y.loc[idx_tr])
        sv_rf = shap.TreeExplainer(rf).shap_values(Xm.loc[idx_te])
        sv_rf = sv_rf[1] if isinstance(sv_rf, list) else (sv_rf[:, :, 1] if sv_rf.ndim == 3 else sv_rf)
        tab["RF"] = np.abs(sv_rf).mean(0)
    for c in list(tab.columns):
        tab[f"rank_{c}"] = tab[c].rank(ascending=False).astype(int)
    print("\nCross-model mean |SHAP| and ranks (medium set, seed 42)\n" + tab.round(4).to_string())
    save(tab.reset_index().rename(columns={"index": "feature"}), "shap_cross_model.csv")

    # --- Part C: rank stability across 30 seeds ------------------------------
    rows = []
    for seed in range(C.N_SEEDS):
        itr, ite = split(df, seed)
        m = make_model("xgb", seed).fit(X.loc[itr], y.loc[itr])
        r = mean_abs_rank(shap_values_xgb(m, X.loc[ite]), X.columns)["rank"]
        rows.append({"seed": seed, **r.to_dict()})
    stab = pd.DataFrame(rows)
    n_ok = int(((stab.priors_count == 1) & (stab.age == 2)).sum())
    print(f"\nprior offence count rank 1 and age rank 2 in {n_ok}/{C.N_SEEDS} seeds")
    save(stab, "shap_rank_stability_seeds.csv")

    # --- Part D: case-level explanations (supplementary) and counterfactuals -
    for case in (C.CASE_A, C.CASE_B):
        verify_case(df, case)
    if HAS_SHAP:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        expl = shap.TreeExplainer(focal)(X)
        for tag, case in (("a", C.CASE_A), ("b", C.CASE_B)):
            pos = X.index.get_loc(case["index"])
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(expl[pos], show=False)
            plt.tight_layout()
            plt.savefig(out(f"waterfall_case_{tag}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"saved: outputs/waterfall_case_{tag}.png")
    # exhaustive integer sweep: smallest change of one feature that flips the prediction
    flips = []
    for tag, case in (("A", C.CASE_A), ("B", C.CASE_B)):
        q = X.loc[[case["index"]]]
        orig = int(focal.predict(q)[0])
        for f in C.INTEGER_FEATURES:
            lo, hi = int(X[f].min()), int(X[f].max())
            cur = int(q[f].iloc[0])
            found = None
            for v in sorted(range(lo, hi + 1), key=lambda v: abs(v - cur)):
                if v == cur:
                    continue
                qq = q.copy()
                qq[f] = float(v)
                if int(focal.predict(qq)[0]) != orig:
                    found = v
                    break
            flips.append({"case": tag, "feature": f, "original": cur, "flip_at": found,
                          "observed_max": hi, "prediction": orig})
    flips = pd.DataFrame(flips)
    print("\nMinimal single-feature changes that flip the prediction (None = no value in the observed range flips it)")
    print(flips.to_string(index=False))
    save(flips, "minimal_flips.csv")

    try:
        import dice_ml
        from dice_ml import Dice
        train_df = pd.concat([X.loc[idx_tr].reset_index(drop=True), y.loc[idx_tr].reset_index(drop=True)], axis=1)
        d = dice_ml.Data(dataframe=train_df, continuous_features=C.INTEGER_FEATURES, outcome_name=C.TARGET)
        exp = Dice(d, dice_ml.Model(model=focal, backend="sklearn", model_type="classifier"), method="random")
        for tag, case in (("A", C.CASE_A), ("B", C.CASE_B)):
            q = X.loc[[case["index"]]]
            orig = int(focal.predict(q)[0])
            cf = exp.generate_counterfactuals(q, total_CFs=10, desired_class="opposite",
                                              features_to_vary=C.INTEGER_FEATURES,
                                              random_seed=C.SEED_DESCRIPTIVE)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            kept = []
            for _, r in cf_df.iterrows():   # project onto the integer domain and re-verify
                feat = r[X.columns].astype(float).copy()
                for f in C.INTEGER_FEATURES:
                    feat[f] = max(0, round(float(feat[f])))
                if int(focal.predict(pd.DataFrame([feat]))[0]) != orig:
                    changed = {f: (int(q[f].iloc[0]), int(feat[f])) for f in X.columns if abs(q[f].iloc[0] - feat[f]) > 1e-9}
                    kept.append({"changes": "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items())})
                if len(kept) >= 3:
                    break
            res = pd.DataFrame(kept)
            print(f"\nCase {tag}: integer-verified counterfactuals\n" + (res.to_string(index=False) if len(res) else "  none"))
            save(res, f"counterfactuals_case_{tag}.csv")
    except ImportError:
        print("\ndice-ml not installed: counterfactuals skipped (pip install dice-ml)")

    # --- Part E: attribution at alpha = 0 vs alpha = 1 -----------------------
    tr, te = df.loc[idx_tr], df.loc[idx_te]
    r_tr, r_te, _ = residualize(tr, te, C.PROXY)
    res = {}
    for a in (0.0, 1.0):
        Xtr, Xte = X.loc[idx_tr].copy(), X.loc[idx_te].copy()
        Xtr[C.PROXY] = mix(tr[C.PROXY], r_tr, a).values
        Xte[C.PROXY] = mix(te[C.PROXY], r_te, a).values
        m = make_model("xgb", C.SEED_DESCRIPTIVE).fit(Xtr, y.loc[idx_tr])
        res[f"alpha={a:.0f}"] = pd.Series(np.abs(shap_values_xgb(m, Xte)).mean(0), index=X.columns)
    cmp = pd.DataFrame(res)
    cmp["rank_alpha=0"] = cmp["alpha=0"].rank(ascending=False).astype(int)
    cmp["rank_alpha=1"] = cmp["alpha=1"].rank(ascending=False).astype(int)
    print("\nGlobal mean |SHAP| before and after residualization\n" + cmp.round(4).to_string())
    save(cmp.reset_index().rename(columns={"index": "feature"}), "shap_alpha0_vs_alpha1.csv")


if __name__ == "__main__":
    main()
