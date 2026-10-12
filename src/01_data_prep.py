"""
01 - Data preparation and sample description (Section 3.1).

Downloads the ProPublica file (once), applies the standard filter,
restricts the sample to the two largest groups and reports the figures
quoted in Section 3.1: group sizes, base rates, and the monthly
recidivism rates that reveal the censoring artifact.

Outputs: data/processed/df_main.csv, outputs/censoring_by_month.csv
"""

import pandas as pd

import config as C
from common import load_compas, save


def main():
    df = load_compas()
    print(f"n = {len(df)}  |  {df[C.PROTECTED].value_counts().to_dict()}")
    print("base rates:", df.groupby(C.PROTECTED)[C.TARGET].mean().round(3).to_dict())

    # Censoring artifact (Section 3.1): monthly recidivism rate and case count
    d = df.copy()
    d["month"] = pd.to_datetime(d["compas_screening_date"]).dt.to_period("M")
    monthly = d.groupby("month").agg(n=(C.TARGET, "size"), rate=(C.TARGET, "mean")).round(3)
    print("\nMonthly recidivism rate (Section 3.1):")
    print(monthly.to_string())
    late = d[pd.to_datetime(d["compas_screening_date"]) >= pd.Timestamp(C.CENSORING_START)]
    print(f"\nscreenings from {C.CENSORING_START}: n = {len(late)} "
          f"({len(late) / len(df):.1%} of the sample), excluded from temporal analyses")
    save(monthly.reset_index(), "censoring_by_month.csv")


if __name__ == "__main__":
    main()
