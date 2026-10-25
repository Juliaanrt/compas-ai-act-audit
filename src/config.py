"""
Central configuration for the COMPAS / EU AI Act audit.

Every script imports its constants from here so that seeds, feature sets
and paths are defined once. Section numbers refer to the manuscript.
"""

from pathlib import Path

# --- Paths (relative to the repository root) --------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/"
    "compas-analysis/master/compas-scores-two-years.csv"
)
COMPAS_FILE = DATA_RAW / "compas-scores-two-years.csv"
DF_MAIN_FILE = DATA_PROCESSED / "df_main.csv"

# --- Sample (Section 3.1) ---------------------------------------------------
GROUPS = ["African-American", "Caucasian"]   # the two largest groups
PROTECTED = "race"
DISADVANTAGED = "African-American"           # coded 1 in the binary indicator
TARGET = "two_year_recid"

# --- Feature sets (Section 3.2) --------------------------------------------
FEATURES_MINIMAL = ["age", "sex"]
FEATURES_MEDIUM = ["age", "sex", "priors_count", "c_charge_degree"]
FEATURES_FULL = FEATURES_MEDIUM + ["juv_fel_count", "juv_misd_count", "juv_other_count"]
FEATURE_SETS = {"Minimal": FEATURES_MINIMAL, "Medium": FEATURES_MEDIUM, "Full": FEATURES_FULL}

# --- Seeds ------------------------------------------------------------------
SEED_DESCRIPTIVE = 42       # single-split descriptive results (Tables 2-4, Figures 1-2)
N_SEEDS = 30                # multi-seed de-biasing design, seeds 0..29 (Section 3.5)
N_SEEDS_WINDOW = 20         # temporal window sensitivity (Table 6)
N_BOOT = 500                # bootstrap resamples for SHAP rank stability (Section 4.5)
TEST_SIZE = 0.2             # 80/20 stratified split throughout

# --- De-biasing experiment (Section 3.5) ------------------------------------
ALPHAS = [round(a * 0.05, 2) for a in range(21)]   # 0.00, 0.05, ..., 1.00
PROXY = "priors_count"      # dominant proxy channel (Section 4.3)
SECOND_PROXY = "age"        # second proxy channel
RECOV_DEPTH = 6             # depth of the recoverability probe (Feldman et al.)

# --- Case-level analysis (Section 4.3, supplementary waterfalls) -----------
# Row labels of the ProPublica CSV. Verified by attribute in 04_shap_audit.py.
CASE_A = {"index": 142, "race": "African-American", "age": 20, "priors_count": 1, "decile_score": 10}
CASE_B = {"index": 568, "race": "Caucasian", "age": 44, "priors_count": 11, "decile_score": 3}
INTEGER_FEATURES = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]

# --- Temporal analysis (Sections 3.1, 4.5) ----------------------------------
CENSORING_START = "2014-04-01"     # screenings from this date on are excluded (Section 3.1)
WINDOW_CUTOFFS = ["2013-07-01", "2013-08-01", "2013-09-01",
                  "2013-10-01", "2013-11-01", "2013-12-01"]

# --- Fairness conventions (Section 3.4) -------------------------------------
DISPARITY_THRESHOLD = 0.1   # disclosed operationalization used in figures
FOUR_FIFTHS = 0.8           # EEOC four-fifths rule for the disparate impact ratio
COMPAS_HIGH_DECILE = 7      # vendor's "high" category
ECE_BINS = 10

# --- Synthetic data (Appendix A) --------------------------------------------
SYN_N = 10_000
SYN_P_GROUP = 0.6
SYN_LAMBDA0 = 2.0
SYN_DELTA = 1.95            # mirrors the offset estimated on COMPAS
SYN_BETA0 = -1.2
SYN_BETA1 = 0.45
SYN_DELTA_SWEEP = [round(1.0 + 0.1 * i, 1) for i in range(21)]   # 1.0 ... 3.0
