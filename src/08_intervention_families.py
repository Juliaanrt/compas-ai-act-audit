"""
08 - Comparison across intervention families (Sections 3.5, 4.5).

On the identical 30-seed design (XGBoost, medium feature set) three
interventions are compared against the untreated model:
    residualization   pre-processing (Section 3.5), alpha = 1
    in-processing     reductions approach of Agarwal et al. under a
                      demographic parity constraint (fairlearn)
    post-processing   group-specific thresholds (fairlearn ThresholdOptimizer)
Recorded per seed: change in DPD and accuracy, and the flip rate (share of
test cases whose prediction changes when every individual's group label is
swapped at scoring time). In-processing has a flip rate of zero by
construction because the fitted model does not receive the group.

Runtime: roughly 5-10 minutes.
Outputs: intervention_families_raw.csv, intervention_families.csv
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient

import config as C
from common import dpd, encode, fmt, load_compas, make_model, paired_summary, save, split

FEATURES = C.FEATURES_MEDIUM


class Float64Probs(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier so that predict_proba returns float64
    (fairlearn's ThresholdOptimizer rejects XGBoost's float32 output)."""
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        self.fitted_ = True
    def fit(self, X, y, **kw): return self
    def predict_proba(self, X): return self.model.predict_proba(np.asarray(X)).astype(np.float64)
    def predict(self, X): return self.model.predict(np.asarray(X))


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    pi = FEATURES.index(C.PROXY)
    rows = []
    for seed in range(C.N_SEEDS):
        itr, ite = split(df, seed)
        tr, te = df.loc[itr], df.loc[ite]
        Xtr, Xte = encode(tr, FEATURES).values, encode(te, FEATURES).values
        ytr, yte = y.loc[itr].values, y.loc[ite].values
        s_tr, s_te = tr[C.PROTECTED].values, te[C.PROTECTED].values
        s_swapped = np.where(s_te == C.DISADVANTAGED, "Caucasian", C.DISADVANTAGED)
        base = make_model("xgb", seed).fit(Xtr, ytr)
        yb = base.predict(Xte)
        dpd0, acc0 = dpd(yte, yb, s_te), accuracy_score(yte, yb)
        r = {"seed": seed, "DPD_base": dpd0, "acc_base": acc0}

        # residualization, alpha = 1 (regression fitted on the training partition)
        g_tr = (s_tr == C.DISADVANTAGED).astype(float)
        g_te = (s_te == C.DISADVANTAGED).astype(float)
        reg = LinearRegression().fit(g_tr.reshape(-1, 1), tr[C.PROXY].values)
        Xtr_r, Xte_r, Xte_s = Xtr.copy(), Xte.copy(), Xte.copy()
        Xtr_r[:, pi] = tr[C.PROXY].values - reg.predict(g_tr.reshape(-1, 1))
        Xte_r[:, pi] = te[C.PROXY].values - reg.predict(g_te.reshape(-1, 1))
        Xte_s[:, pi] = te[C.PROXY].values - reg.predict((1 - g_te).reshape(-1, 1))
        mr = make_model("xgb", seed).fit(Xtr_r, ytr)
        yr = mr.predict(Xte_r)
        r.update(resid_dDPD=dpd(yte, yr, s_te) - dpd0, resid_dacc=accuracy_score(yte, yr) - acc0,
                 resid_flip=(yr != mr.predict(Xte_s)).mean())

        # in-processing
        eg = ExponentiatedGradient(make_model("xgb", seed), constraints=DemographicParity())
        eg.fit(Xtr, ytr, sensitive_features=s_tr)
        ye = np.asarray(eg.predict(Xte, random_state=seed))
        r.update(inproc_dDPD=dpd(yte, ye, s_te) - dpd0, inproc_dacc=accuracy_score(yte, ye) - acc0, inproc_flip=0.0)

        # post-processing
        to = ThresholdOptimizer(estimator=Float64Probs(base), constraints="demographic_parity",
                                predict_method="predict_proba", prefit=True)
        to.fit(Xtr, ytr, sensitive_features=s_tr)
        yp = np.asarray(to.predict(Xte, sensitive_features=s_te, random_state=seed))
        yps = np.asarray(to.predict(Xte, sensitive_features=s_swapped, random_state=seed))
        r.update(post_dDPD=dpd(yte, yp, s_te) - dpd0, post_dacc=accuracy_score(yte, yp) - acc0,
                 post_flip=(yp != yps).mean())
        rows.append(r)
        print(f"seed {seed} done")

    raw = pd.DataFrame(rows)
    save(raw, "intervention_families_raw.csv")
    summ = []
    for fam in ["resid", "inproc", "post"]:
        for q in ["dDPD", "dacc"]:
            d = paired_summary(raw[f"{fam}_{q}"])
            d.update(family=fam, quantity=q)
            summ.append(d)
            print(f"  {fam:<7} {q:<5}", fmt(d))
        print(f"  {fam:<7} flip  {raw[f'{fam}_flip'].mean():.3f}")
    save(pd.DataFrame(summ), "intervention_families.csv")


if __name__ == "__main__":
    main()
