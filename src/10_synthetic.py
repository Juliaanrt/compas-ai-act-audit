"""
10 - Mechanism isolation on synthetic data (Section 4.7, Table 8,
Figure 4, Appendix A).

Data-generating process (Appendix A.1): group ~ Bernoulli(0.6); count ~
Poisson(lambda0 + delta * group); two standard-normal nuisance features
independent of group and outcome; y ~ Bernoulli(sigmoid(beta0 + beta1 *
count)). Each condition is replicated over 30 seeds with an 80/20 split
stratified on y; the offset is estimated on the training partition.

Conditions (Appendix A.2), all with the same true delta = 1.95:
    A  count - delta_hat * group                       (fractional offset)
    B  count - round(delta_hat) * group                (integer offset)
    C  continuous proxy N(lambda, sqrt(lambda)), shifted by delta_hat
    D  round(count - delta_hat * group)                (rounded after shift)
    E  clip(count - round(delta_hat) * group, 0, inf)  (B, clipped to support)

Recorded: change in DPD (XGBoost, logistic regression) and accuracy
between the untransformed and the transformed feature; recoverability of
the group from the transformed feature alone (depth-6 and unrestricted
tree, held-out); test observations below the original support.

Offset sweep (Appendix A.3): true delta from 1.0 to 3.0, 30 seeds each,
recoverability and fractional part of delta_hat (Figure 4).

Outputs: table8_synthetic.csv, synthetic_raw.csv, delta_sweep.csv, figure4_synthetic.png
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config as C
from common import make_model, out, recoverability, save


def generate(seed, delta=C.SYN_DELTA, continuous=False):
    rng = np.random.default_rng(seed)
    g = rng.binomial(1, C.SYN_P_GROUP, C.SYN_N)
    lam = C.SYN_LAMBDA0 + delta * g
    x = rng.normal(lam, np.sqrt(lam)) if continuous else rng.poisson(lam).astype(float)
    z1, z2 = rng.normal(size=C.SYN_N), rng.normal(size=C.SYN_N)
    y = rng.binomial(1, 1 / (1 + np.exp(-(C.SYN_BETA0 + C.SYN_BETA1 * x))))
    return pd.DataFrame(dict(g=g, x=x, z1=z1, z2=z2, y=y))


def transform(xtr, gtr, xte, gte, cond):
    d = xtr[gtr == 1].mean() - xtr[gtr == 0].mean()       # least-squares offset on the training partition
    dd = np.round(d) if cond in ("B", "E") else d
    ptr, pte = xtr - dd * gtr, xte - dd * gte
    if cond == "D":
        ptr, pte = np.round(ptr), np.round(pte)
    if cond == "E":
        ptr, pte = np.clip(ptr, 0, None), np.clip(pte, 0, None)
    return ptr, pte, d


def dpd_np(yhat, g):
    return abs(yhat[g == 1].mean() - yhat[g == 0].mean())


def run_condition(cond, seed):
    d = generate(seed, continuous=(cond == "C"))
    tr, te = train_test_split(d, test_size=C.TEST_SIZE, random_state=seed, stratify=d.y)
    xtr, xte, gtr, gte = tr.x.values, te.x.values, tr.g.values, te.g.values
    ptr, pte, delta_hat = transform(xtr, gtr, xte, gte, cond)
    r = dict(cond=cond, seed=seed, delta_hat=delta_hat,
             rec_base=recoverability(xtr, gtr, xte, gte, C.RECOV_DEPTH, seed),
             rec_depth6=recoverability(ptr, gtr, pte, gte, C.RECOV_DEPTH, seed),
             rec_unrestricted=recoverability(ptr, gtr, pte, gte, None, seed),
             out_of_range=int((pte < 0).sum()))
    Z_tr, Z_te = tr[["z1", "z2"]].values, te[["z1", "z2"]].values
    for m in ("xgb", "lr"):
        m0 = make_model(m, seed).fit(np.c_[xtr, Z_tr], tr.y)
        m1 = make_model(m, seed).fit(np.c_[ptr, Z_tr], tr.y)
        y0, y1 = m0.predict(np.c_[xte, Z_te]), m1.predict(np.c_[pte, Z_te])
        r[f"{m}_dDPD"] = dpd_np(y1, gte) - dpd_np(y0, gte)
        r[f"{m}_dacc"] = accuracy_score(te.y, y1) - accuracy_score(te.y, y0)
    return r


def main():
    # --- Table 8 -----------------------------------------------------------------
    rows = [run_condition(c, s) for c in "ABCDE" for s in range(C.N_SEEDS)]
    raw = pd.DataFrame(rows)
    save(raw, "synthetic_raw.csv")
    t8 = raw.groupby("cond").agg(dDPD_XGB=("xgb_dDPD", "mean"), dDPD_LR=("lr_dDPD", "mean"),
                                 dAcc_XGB=("xgb_dacc", "mean"), rec_depth6=("rec_depth6", "mean"),
                                 rec_unrestricted=("rec_unrestricted", "mean"), out_of_range=("out_of_range", "mean"),
                                 rec_baseline=("rec_base", "mean"), delta_hat=("delta_hat", "mean")).round(3)
    print("Table 8\n" + t8.to_string())
    print(f"grid channel (A - B, depth 6): {t8.loc['A', 'rec_depth6'] - t8.loc['B', 'rec_depth6']:.3f}; "
          f"range channel (B - E): {t8.loc['B', 'rec_depth6'] - t8.loc['E', 'rec_depth6']:.3f}")
    save(t8.reset_index(), "table8_synthetic.csv")

    # --- Offset sweep (Figure 4) -------------------------------------------------
    rows = []
    for delta in C.SYN_DELTA_SWEEP:
        for seed in range(C.N_SEEDS):
            d = generate(seed, delta=delta)
            tr, te = train_test_split(d, test_size=C.TEST_SIZE, random_state=seed, stratify=d.y)
            ptr, pte, dh = transform(tr.x.values, tr.g.values, te.x.values, te.g.values, "A")
            rows.append(dict(delta=delta, seed=seed, delta_hat=dh, frac=abs(dh - round(dh)),
                             rec_raw=recoverability(tr.x.values, tr.g.values, te.x.values, te.g.values, C.RECOV_DEPTH, seed),
                             rec_depth6=recoverability(ptr, tr.g.values, pte, te.g.values, C.RECOV_DEPTH, seed),
                             rec_unrestricted=recoverability(ptr, tr.g.values, pte, te.g.values, None, seed)))
    sweep = pd.DataFrame(rows)
    save(sweep, "delta_sweep.csv")
    g = sweep.groupby("delta").agg(rec_unrestricted=("rec_unrestricted", "mean"), rec_depth6=("rec_depth6", "mean"),
                                   rec_raw=("rec_raw", "mean"), frac=("frac", "mean"),
                                   frac_lo=("frac", lambda s: np.percentile(s, 2.5)),
                                   frac_hi=("frac", lambda s: np.percentile(s, 97.5)))
    print("\nOffset sweep\n" + g.round(3).to_string())
    print(f"smallest fractional part of delta_hat: {sweep.frac.min():.4f}; "
          f"minimum unrestricted recoverability: {sweep.rec_unrestricted.min():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
        a1.plot(g.index, g.rec_unrestricted, "o-", color="black", ms=3.5, label="transformed, unrestricted tree")
        a1.plot(g.index, g.rec_depth6, "s--", color="0.35", ms=3.5, label="transformed, depth-6 tree")
        a1.plot(g.index, g.rec_raw, "^:", color="0.6", ms=3.5, label="untransformed count")
        for x in (1, 2, 3):
            a1.axvline(x, color="0.85", lw=0.8, zorder=0)
            a2.axvline(x, color="0.85", lw=0.8, zorder=0)
        a1.set_xlabel("true group offset δ")
        a1.set_ylabel("recoverability (AUC)")
        a1.set_ylim(0.5, 1.03)
        a1.set_title("(a) Attribute recoverability", fontsize=9, loc="left")
        a1.legend(frameon=False, fontsize=7.5)
        a2.fill_between(g.index, g.frac_lo, g.frac_hi, color="0.85", lw=0)
        a2.plot(g.index, g.frac, "o-", color="black", ms=3.5)
        a2.set_xlabel("true group offset δ")
        a2.set_ylabel("fractional part of δ̂")
        a2.set_ylim(-0.02, 0.55)
        a2.set_title("(b) Fractional part of the estimate", fontsize=9, loc="left")
        for ax in (a1, a2):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out("figure4_synthetic.png"), dpi=600)
        print("saved: outputs/figure4_synthetic.png")
    except Exception as e:
        print("figure skipped:", e.__class__.__name__)


if __name__ == "__main__":
    main()
