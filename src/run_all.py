"""Runs the full pipeline in order. Total runtime roughly 1.5-2 hours."""

import runpy
import sys
from pathlib import Path

SCRIPTS = ["01_data_prep", "02_baseline_models", "03_fairness_focal", "04_shap_audit",
           "05_proxy_statistics", "06_judge_baseline", "07_debiasing_multiseed",
           "08_intervention_families", "09_temporal_robustness", "10_synthetic",
           "11_mechanism_check"]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    for s in SCRIPTS:
        print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)
        runpy.run_path(str(here / f"{s}.py"), run_name="__main__")
