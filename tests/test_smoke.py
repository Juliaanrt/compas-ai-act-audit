"""
Fast end-to-end check of the shared helpers on the real sample.

Runs in about a minute and verifies the invariants the manuscript relies
on: sample size, base rates, exact zero correlation after residualization,
disjoint value grids, and the direction of the recoverability effect on
the seed-42 split. Run with:  python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
from common import (encode, group_indicator, load_compas, make_model,
                    recoverability, residualize, split)


@pytest.fixture(scope="module")
def df():
    return load_compas()


def test_sample(df):
    assert len(df) == 5278
    counts = df[C.PROTECTED].value_counts()
    assert counts["African-American"] == 3175 and counts["Caucasian"] == 2103
    rates = df.groupby(C.PROTECTED)[C.TARGET].mean()
    assert abs(rates["African-American"] - 0.523) < 0.001
    assert abs(rates["Caucasian"] - 0.391) < 0.001


def test_residualization_removes_linear_correlation(df):
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    tr, te = df.loc[itr], df.loc[ite]
    r_tr, _, delta = residualize(tr, te, C.PROXY)
    assert 1.8 < delta < 2.1
    corr = np.corrcoef(r_tr, group_indicator(tr))[0, 1]
    assert abs(corr) < 1e-10


def test_groups_land_on_disjoint_grids(df):
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    tr, te = df.loc[itr], df.loc[ite]
    _, r_te, _ = residualize(tr, te, C.PROXY)
    g = group_indicator(te).values
    frac_aa = set(np.round(r_te[g == 1] % 1, 6))
    frac_ca = set(np.round(r_te[g == 0] % 1, 6))
    assert len(frac_aa) == 1 and len(frac_ca) == 1
    assert frac_aa.isdisjoint(frac_ca)


def test_recoverability_rises(df):
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    tr, te = df.loc[itr], df.loc[ite]
    r_tr, r_te, _ = residualize(tr, te, C.PROXY)
    gtr, gte = group_indicator(tr).values, group_indicator(te).values
    before = recoverability(tr[C.PROXY], gtr, te[C.PROXY], gte)
    after = recoverability(r_tr, gtr, r_te, gte)
    assert before < 0.7 and after > 0.9


def test_focal_model_fits(df):
    y = df[C.TARGET].astype(int)
    X = encode(df, C.FEATURES_FULL)
    itr, ite = split(df, C.SEED_DESCRIPTIVE)
    m = make_model("xgb", C.SEED_DESCRIPTIVE).fit(X.loc[itr], y.loc[itr])
    acc = (m.predict(X.loc[ite]) == y.loc[ite]).mean()
    assert abs(acc - 0.655) < 0.005
