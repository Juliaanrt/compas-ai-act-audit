# Racial Bias, Explainability, and EU AI Act Compliance: A SHAP-Based Empirical Analysis of Recidivism Prediction on the COM-PAS Dataset — analysis code

Reproduction code for *Racial Bias, Explainability, and EU AI Act Compliance: A SHAP-Based Empirical Analysis of Recidivism Prediction on the COM-PAS Dataset* (Julian Reiter, FOM Hochschule für Ökonomie &
Management).

## Setup

Python 3.11–3.13. Install the pinned dependencies:

```
pip install -r requirements.txt
```

The ProPublica COMPAS file is downloaded automatically on first use into
`data/raw/` (~ 2 MB, public domain). No other data are needed.

`numpy` is pinned below 2.3 because `shap` depends on `numba`, which does
not support newer NumPy releases. `shap` and `dice-ml` are optional: the
scripts fall back to XGBoost's built-in TreeSHAP and skip the waterfall
plots and counterfactuals if they are missing.

## Running

Each script is self-contained and writes its results to `outputs/`.
Run from the repository root:

```
python src/01_data_prep.py
python src/02_baseline_models.py
...
python src/11_mechanism_check.py
```

or everything in order with `python src/run_all.py` (about 1.5–2 hours;
`07` and `09` dominate). Scripts `07` and `08` reuse their raw CSV if it
already exists in `outputs/`; delete it to recompute.

## Map from manuscript to code

| Manuscript | Script | Output |
|---|---|---|
| 3.1 sample, censoring artifact | `01_data_prep.py` | `censoring_by_month.csv` |
| 4.1 Table 2 | `02_baseline_models.py` | `table2_performance.csv` |
| 4.2 Table 3, Figure 1, matched marking rates, calibration | `03_fairness_focal.py` | `table3_fairness.csv`, `figure1_threshold_fpr.png`, `matched_marking_rates.csv`, `ece_gap_seeds.csv` |
| 4.3 Figure 2, cross-model and cross-seed robustness, case-level explanation; 4.4 attribution at α = 0 vs 1 | `04_shap_audit.py` | `figure2_shap_summary.png`, `shap_cross_model.csv`, `shap_rank_stability_seeds.csv`, `shap_alpha0_vs_alpha1.csv`; supplementary: `waterfall_case_a/b.png`, `minimal_flips.csv`, `counterfactuals_case_*.csv` |
| 4.3 proxy correlations, subgroup structure | `05_proxy_statistics.py` | `proxy_statistics.csv`, `intersectional_groups.csv` |
| 4.2 Table 4, 5.3 detention length | `06_judge_baseline.py` | `table4_three_way.csv` |
| 3.5 / 4.4 Table 5, Figure 3, controls, dose-response, recoverability, age residualization, calibration after residualization; 4.7 integer offset on COMPAS (`resid_round`) | `07_debiasing_multiseed.py` | `table5.csv`, `debiasing_controls.csv`, `dose_response.csv`, `recoverability_summary.csv`, `age_residualization.csv`, `figure3_frontier.png` |
| 4.5 in-/post-processing, flip rate | `08_intervention_families.py` | `intervention_families.csv` |
| 4.5 Table 6, SHAP ranks per window, bootstrap | `09_temporal_robustness.py` | `table6_window_sensitivity.csv`, `shap_windows.csv`, `shap_bootstrap.csv` |
| 4.7 Table 8, Figure 4, Appendix A | `10_synthetic.py` | `table8_synthetic.csv`, `delta_sweep.csv`, `figure4_synthetic.png` |
| 4.4 booster dumps, grid signature | `11_mechanism_check.py` | `mechanism_check.csv` |

Table 9 summarizes hypothesis decisions drawn from the outputs of `07`, `07` controls and `10`.

A fast consistency check of the shared helpers (sample size, zero
correlation after residualization, disjoint grids, recoverability) runs
with `python -m pytest tests/` in about a minute.

Shared operations (filter, encoding, stratified split, residualization on
the training partition, fairness metrics, ECE, recoverability probe,
paired bootstrap/Wilcoxon statistics) live in `src/common.py`; all
constants (seeds, feature sets, α grid, cutoff dates, synthetic
parameters) in `src/config.py`.

## Design choices fixed in `config.py`

- Seed 42 for every single-split descriptive result; seeds 0–29 for the
  multi-seed design; seeds 0–19 per temporal window.
- Library-default hyper-parameters, no tuning (Section 3.2).
- α ∈ {0, 0.05, …, 1}; the residualization regression is fitted on the
  training partition of each split only.
- Recoverability probe: decision tree of depth 6 (plus unrestricted tree
  and logistic regression as bounds), fitted on the training partition,
  scored on the test partition.
- Screenings from 2014-04-01 are excluded from all temporal analyses.
