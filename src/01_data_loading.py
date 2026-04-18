"""Step 1: load the three SSLC 2022 Stata files and run basic audits.

Reads RT002 (persons), RT001 (housing), and RT140 (food) from `data/`.
Prints schema / head / dtype summaries, checks the columns the downstream
scripts depend on, writes schema files under `appendix_exports/`, and
computes the national P20 food-expenditure cutoff used as the economic
deprivation threshold.

Usage:
    python src/01_data_loading.py
"""

from __future__ import annotations

import pandas as pd
import pyreadstat

from utils import (
    FOOD_PATH,
    HOUSING_PATH,
    OUT_DIR,
    PERSONS_PATH,
    h,
    missing_table,
    must_have,
    short_info,
    weighted_quantile,
)


def load_stata(path) -> pd.DataFrame:
    """Load a .dta file with pyreadstat, preserving value labels where useful."""
    df, meta = pyreadstat.read_dta(str(path), apply_value_formats=False)
    print(f"[load] {path.name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def main() -> None:
    h("1. LOAD SSLC 2022 STATA FILES")

    persons = load_stata(PERSONS_PATH)
    housing = load_stata(HOUSING_PATH)
    food = load_stata(FOOD_PATH)

    short_info(persons, "persons")
    short_info(housing, "housing")
    short_info(food, "food")

    h("2. REQUIRED COLUMN CHECKS")

    persons_required = [
        "hhid",
        "pid",
        "weight",
        "q01_01",  # age
        "q01_03",  # sex
        "estrato",  # region / stratum
        "q04_05",  # highest education level completed
        "q05_02",  # worked last week (employment)
    ]
    housing_required = ["hhid", "weight", "q13_17"]  # main lighting source
    food_required = ["hhid", "q140_06"]  # per-item expenditure

    must_have(persons, persons_required, "persons")
    must_have(housing, housing_required, "housing")
    must_have(food, food_required, "food")

    h("3. MISSINGNESS AUDIT")

    for name, df, cols in [
        ("persons", persons, persons_required),
        ("housing", housing, housing_required),
        ("food", food, food_required),
    ]:
        tbl = missing_table(df, cols)
        tbl.to_csv(OUT_DIR / f"{name}_missingness.csv", index=False)
        print(f"\n[{name}] missingness:")
        print(tbl.to_string(index=False))

    h("4. NATIONAL P20 FOOD CUTOFF (ECONOMIC DEPRIVATION THRESHOLD)")

    # Sum food expenditure to household level, then attach household weights
    # from RT001 (housing) because RT140 is item-level.
    hh_food = (
        food.dropna(subset=["hhid", "q140_06"])
        .groupby("hhid", as_index=False)["q140_06"]
        .sum()
        .rename(columns={"q140_06": "food_exp"})
    )
    hh_food = hh_food.merge(housing[["hhid", "weight"]], on="hhid", how="left")
    hh_food = hh_food.dropna(subset=["weight"])

    p20 = weighted_quantile(hh_food["food_exp"], 0.20, hh_food["weight"])
    print(f"Weighted P20 of household food expenditure = SRD {p20:,.2f}")
    print(f"Households below P20: {(hh_food['food_exp'] <= p20).sum():,} "
          f"of {len(hh_food):,}")

    # Persist for downstream steps
    hh_food.to_csv(OUT_DIR / "hh_food_expenditure.csv", index=False)
    with open(OUT_DIR / "p20_cutoff.txt", "w", encoding="utf-8") as f:
        f.write(f"{p20:.6f}\n")

    print(f"\n[done] outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
