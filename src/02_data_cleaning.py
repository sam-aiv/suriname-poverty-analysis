"""Step 2: key uniqueness, within-household audits, and merges.

Checks that person and household keys are unique at their respective
levels, audits how much key variables vary within a household, merges
RT002 (persons) with RT001 (housing) on `hhid`, attaches the
household-level food expenditure from step 1, and encodes `region`
(label) and `is_female` (binary) for use in later scripts.

Outputs:
    appendix_exports/persons_clean.parquet
    appendix_exports/hh_clean.parquet
"""

from __future__ import annotations

import pandas as pd
import pyreadstat

from utils import (
    FOOD_PATH,
    HOUSING_PATH,
    OUT_DIR,
    PERSONS_PATH,
    REGION_MAP,
    h,
)


def load(path) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(str(path), apply_value_formats=False)
    return df


def main() -> None:
    h("1. RELOAD RAW FILES")
    persons = load(PERSONS_PATH)
    housing = load(HOUSING_PATH)
    food = load(FOOD_PATH)

    h("2. KEY UNIQUENESS")

    n_persons = len(persons)
    n_unique_pid = persons[["hhid", "pid"]].drop_duplicates().shape[0]
    print(f"persons: {n_persons:,} rows, {n_unique_pid:,} unique (hhid,pid) — "
          f"{'OK' if n_persons == n_unique_pid else 'DUPLICATE KEYS'}")

    n_housing = len(housing)
    n_unique_hh = housing["hhid"].nunique()
    print(f"housing: {n_housing:,} rows, {n_unique_hh:,} unique hhid — "
          f"{'OK' if n_housing == n_unique_hh else 'DUPLICATE KEYS'}")

    h("3. WITHIN-HOUSEHOLD VARIATION AUDIT")

    # How often do key person-level variables actually vary within a household?
    audit_cols = ["q04_05", "q05_02", "q01_01", "q01_03"]
    audit = (
        persons.groupby("hhid")[audit_cols]
        .nunique()
        .reset_index()
        .rename(columns={c: f"nunique_{c}" for c in audit_cols})
    )
    summary = pd.DataFrame(
        {
            "mean_nunique": audit[[c for c in audit.columns if c.startswith("nunique_")]].mean(),
            "pct_hh_varying": (
                audit[[c for c in audit.columns if c.startswith("nunique_")]] > 1
            ).mean() * 100,
        }
    )
    print(summary)
    summary.to_csv(OUT_DIR / "within_hh_variation.csv")

    h("4. REGION + GENDER RECODES")

    persons["region"] = persons["estrato"].map(REGION_MAP)
    # q01_03: 1 male, 2 female in the SSLC codebook
    persons["is_female"] = (persons["q01_03"] == 2).astype(int)
    unmapped = persons["region"].isna().sum()
    if unmapped:
        print(f"[warn] {unmapped:,} persons had an unmapped `estrato` value")

    h("5. MERGE PERSONS <-> HOUSING")

    merge_cols = ["hhid", "q13_17"]  # lighting source
    before = len(persons)
    persons = persons.merge(housing[merge_cols], on="hhid", how="left")
    matched = persons["q13_17"].notna().sum()
    print(f"persons rows: {before:,} -> {len(persons):,}; matched to housing: {matched:,}")

    h("6. ATTACH HOUSEHOLD FOOD EXPENDITURE")

    hh_food = (
        food.dropna(subset=["hhid", "q140_06"])
        .groupby("hhid", as_index=False)["q140_06"]
        .sum()
        .rename(columns={"q140_06": "food_exp"})
    )
    persons = persons.merge(hh_food, on="hhid", how="left")
    print(f"persons with food_exp: {persons['food_exp'].notna().sum():,}")

    h("7. WRITE CLEAN FRAMES")

    persons.to_parquet(OUT_DIR / "persons_clean.parquet", index=False)

    housing["region"] = housing["estrato"].map(REGION_MAP) if "estrato" in housing else None
    housing = housing.merge(hh_food, on="hhid", how="left")
    housing.to_parquet(OUT_DIR / "hh_clean.parquet", index=False)

    print(f"\n[done] persons_clean: {persons.shape} | hh_clean: {housing.shape}")


if __name__ == "__main__":
    main()
