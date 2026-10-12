"""
05 - Proxy statistics and subgroup structure (Sections 3.1, 4.3).

Part A  Pearson correlation of the two proxy channels with the group
        indicator; group medians; Mann-Whitney tests; prior counts among
        actual recidivists.
Part B  Age distribution by group (data quality property, Section 3.1).
Part C  Intersectional view (race x sex x age band) of the focal model:
        FPR and dominant SHAP feature per subgroup (Section 4.3, last
        paragraph).

Outputs: proxy_statistics.csv, intersectional_groups.csv
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from fairlearn.metrics import false_positive_rate

import config as C
from common import by_group, encode, group_indicator, load_compas, make_model, save, shap_values_xgb, split


def main():
    df = load_compas()
    g = group_indicator(df)
    aa, ca = df[df[C.PROTECTED] == C.DISADVANTAGED], df[df[C.PROTECTED] == "Caucasian"]

    # --- Part A / B --------------------------------------------------------------
    rows = []
    for feat in (C.PROXY, C.SECOND_PROXY):
        r, p = stats.pearsonr(df[feat], g)
        u, pu = stats.mannwhitneyu(aa[feat], ca[feat], alternative="two-sided")
        rows.append({"feature": feat, "pearson_r": r, "p_r": p,
                     "median_AA": aa[feat].median(), "median_CA": ca[feat].median(),
                     "mean_AA": aa[feat].mean(), "mean_CA": ca[feat].mean(), "p_mannwhitney": pu})
    rec_aa, rec_ca = aa[aa[C.TARGET] == 1][C.PROXY], ca[ca[C.TARGET] == 1][C.PROXY]
    u, pu = stats.mannwhitneyu(rec_aa, rec_ca, alternative="two-sided")
    rows.append({"feature": f"{C.PROXY} | recidivists only", "median_AA": rec_aa.median(),
                 "median_CA": rec_ca.median(), "p_mannwhitney": pu})
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    save(tab, "proxy_statistics.csv")

    # --- Part C: intersectional groups -------------------------------------------
    d = df.copy()
    d["age_band"] = pd.cut(d["age"], bins=[0, 25, 40, 200], labels=["<=25", "26-40", ">40"])
    d["group"] = d[C.PROTECTED] + " | " + d["sex"] + " | " + d["age_band"].astype(str)
    y = d[C.TARGET].astype(int)
    X = encode(d, C.FEATURES_FULL)
    itr, ite = split(d, C.SEED_DESCRIPTIVE)
    m = make_model("xgb", C.SEED_DESCRIPTIVE).fit(X.loc[itr], y.loc[itr])
    yhat = m.predict(X.loc[ite])
    sv = pd.DataFrame(shap_values_xgb(m, X.loc[ite]), columns=X.columns, index=ite)
    grp = d.loc[ite, "group"]
    fpr = by_group(y.loc[ite], yhat, grp, false_positive_rate)
    counts = grp.value_counts()
    shap_g = sv.abs().groupby(grp).mean()
    res = pd.DataFrame({"n_test": counts, "FPR": fpr,
                        "shap_age": shap_g["age"], "shap_priors": shap_g["priors_count"]})
    res["dominant"] = np.where(res.shap_age > res.shap_priors, "age", "priors_count")
    res = res[res.n_test >= 30].sort_values("FPR", ascending=False).round(3)
    print("\nIntersectional groups (test split, n >= 30)\n" + res.to_string())
    save(res.reset_index().rename(columns={"index": "group"}), "intersectional_groups.csv")


if __name__ == "__main__":
    main()
