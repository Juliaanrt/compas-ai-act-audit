"""
06 - Exploratory human baseline (Section 4.2, Table 4; Section 5.3).

Pretrial detention length above the median proxies the judicial decision.
The three systems are compared on the seed-42 test cases that have a
valid detention record. DIR uses the ratio of positive prediction rates.

Outputs: table4_three_way.csv
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

import config as C
from common import (disparate_impact_ratio, dpd, encode, eod, fpr_gap, load_compas,
                    make_model, save, split)


def jail_days(df):
    d = df.copy()
    d["jail_days"] = (pd.to_datetime(d["c_jail_out"], errors="coerce")
                      - pd.to_datetime(d["c_jail_in"], errors="coerce")).dt.days
    return d.dropna(subset=["jail_days"]).query("jail_days >= 0")


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    X = encode(df, C.FEATURES_FULL)
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    model = make_model("xgb", C.SEED_DESCRIPTIVE).fit(X.loc[itr], y.loc[itr])

    dj = jail_days(df)
    median = dj["jail_days"].median()
    valid = pd.Index(ite).intersection(dj.index)
    yv, sv = y.loc[valid], df.loc[valid, C.PROTECTED]
    preds = {"Judge (detention proxy)": (dj.loc[valid, "jail_days"] > median).astype(int).values,
             f"COMPAS (decile >= {C.COMPAS_HIGH_DECILE})": (df.loc[valid, "decile_score"] >= C.COMPAS_HIGH_DECILE).astype(int).values,
             "XGBoost + SHAP": model.predict(X.loc[valid])}
    rows = []
    for label, yhat in preds.items():
        dir_ = disparate_impact_ratio(yhat, sv)
        rows.append({"System": label, "DPD": dpd(yv, yhat, sv), "EOD": eod(yv, yhat, sv),
                     "FPR gap": fpr_gap(yv, yhat, sv), "DIR": dir_,
                     "4/5 rule": "PASS" if dir_ >= C.FOUR_FIFTHS else "FAIL"})
    tab = pd.DataFrame(rows).round(3)
    print(f"n = {len(valid)} test cases with valid detention record\n" + tab.to_string(index=False))
    save(tab, "table4_three_way.csv")

    a = dj[dj[C.PROTECTED] == C.DISADVANTAGED]["jail_days"]
    c = dj[dj[C.PROTECTED] == "Caucasian"]["jail_days"]
    _, p = stats.mannwhitneyu(a, c, alternative="two-sided")
    print(f"\nDetention length: mean AA {a.mean():.1f} vs CA {c.mean():.1f} days "
          f"(+{(a.mean() / c.mean() - 1):.0%}), Mann-Whitney p = {p:.2e}")


if __name__ == "__main__":
    main()
