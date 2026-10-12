"""
11 - Mechanism check on the fitted trees (Section 4.4).

Verifies on the seed-42 split that the residualization places the two
groups on disjoint value grids and that the booster places split
thresholds between the grids:
    A  split thresholds on the proxy feature in the booster dump
    B  grid signature: fractional parts of p(alpha) per group
    C  recoverability of the group from p(alpha) alone (depth-6 tree)

Outputs: mechanism_check.csv
"""

import re

import numpy as np
import pandas as pd

import config as C
from common import encode, group_indicator, load_compas, make_model, mix, recoverability, residualize, save, split

ALPHAS = [0.0, 0.05, 0.5, 1.0]


def thresholds(model, feature):
    pat = re.compile(r"\[" + re.escape(feature) + r"<([-0-9eE.+]+)\]")
    return np.array([float(v) for t in model.get_booster().get_dump() for v in pat.findall(t)])


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    tr, te = df.loc[itr], df.loc[ite]
    r_tr, r_te, delta = residualize(tr, te, C.PROXY)
    gtr, gte = group_indicator(tr).values, group_indicator(te).values
    rows = []
    for alpha in ALPHAS:
        p_tr, p_te = mix(tr[C.PROXY], r_tr, alpha), mix(te[C.PROXY], r_te, alpha)
        Xtr, Xte = encode(tr, C.FEATURES_FULL), encode(te, C.FEATURES_FULL)
        Xtr[C.PROXY], Xte[C.PROXY] = p_tr.values, p_te.values
        m = make_model("xgb", C.SEED_DESCRIPTIVE).fit(Xtr, y.loc[itr])
        th = thresholds(m, C.PROXY)
        aa, ca = p_te[gte == 1], p_te[gte == 0]
        off_aa = set(np.round(aa - np.floor(aa), 6))
        off_ca = set(np.round(ca - np.floor(ca), 6))
        rows.append({"alpha": alpha, "n_splits_on_proxy": len(th), "n_unique_thresholds": len(np.unique(np.round(th, 4))),
                     "first_thresholds": np.sort(np.unique(np.round(th, 3)))[:8].tolist(),
                     "fractional_offsets_AA": sorted(off_aa)[:3], "fractional_offsets_CA": sorted(off_ca)[:3],
                     "grids_disjoint": len(off_aa & off_ca) == 0,
                     "recoverability_depth6": recoverability(p_tr, gtr, p_te, gte, C.RECOV_DEPTH, C.SEED_DESCRIPTIVE)})
    res = pd.DataFrame(rows)
    print(f"delta_hat = {delta:.4f}\n" + res.to_string(index=False))
    save(res, "mechanism_check.csv")


if __name__ == "__main__":
    main()
