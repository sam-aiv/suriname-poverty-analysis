"""Step 3: build the four deprivation indicators and the MPI.

Constructs, at person level:
    education_pov   — did not complete at least lower-secondary schooling
    employment_pov  — not working last week (for working-age individuals)
    electricity_pov — household main lighting source is not the grid
    econ_pov        — household food expenditure <= national P20

Then builds:
    score       — unweighted sum of the four binaries (0..4)
    c_score     — Alkire-Foster c(k) with equal weights and K = 1/3
    poor_af     — AF poor flag (c_score > K)
    poor_rel_region — poor relative to the household's region median score

Reports Headcount (H), Average intensity (A), and MPI = H * A by region,
plus a PCA sanity check showing that the four indicators are positively
but only modestly correlated with their first principal component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import (
    ELECTRIC_GRID,
    OUT_DIR,
    REGION_ORDER,
    h,
    weighted_quantile,
    weighted_rate,
)

# Alkire-Foster poverty cutoff (share of weighted deprivations).
K_CUTOFF = 1.0 / 3.0


def main() -> None:
    h("1. LOAD CLEAN PERSONS FRAME")
    persons = pd.read_parquet(OUT_DIR / "persons_clean.parquet")
    print(f"persons: {persons.shape}")

    h("2. BUILD BINARY DEPRIVATIONS")

    # Education: q04_05 coded so that >= 3 means lower-secondary completed
    # or above (see codebook). Missing education -> deprived by convention.
    persons["education_pov"] = (persons["q04_05"].fillna(0) < 3).astype(int)

    # Employment: q05_02 == 1 means worked last week; restrict to 15-64.
    persons["working_age"] = persons["q01_01"].between(15, 64).astype(int)
    persons["employment_pov"] = np.where(
        persons["working_age"] == 1,
        (persons["q05_02"] != 1).astype(int),
        np.nan,
    )

    # Electricity: grid = not deprived, anything else = deprived
    persons["electricity_pov"] = (~persons["q13_17"].isin(ELECTRIC_GRID)).astype(int)
    persons.loc[persons["q13_17"].isna(), "electricity_pov"] = np.nan

    # Economic: household food expenditure <= national P20
    p20 = weighted_quantile(persons["food_exp"], 0.20, persons["weight"])
    print(f"P20 food expenditure cutoff: SRD {p20:,.2f}")
    persons["econ_pov"] = (persons["food_exp"] <= p20).astype(int)
    persons.loc[persons["food_exp"].isna(), "econ_pov"] = np.nan

    h("3. COMPOSITE SCORE + AF MPI")

    indicators = ["education_pov", "employment_pov", "electricity_pov", "econ_pov"]
    persons["score"] = persons[indicators].sum(axis=1, min_count=1)
    persons["c_score"] = persons[indicators].mean(axis=1)
    persons["poor_af"] = (persons["c_score"] > K_CUTOFF).astype(int)

    # Regional relative poverty: poor if score > region weighted median
    region_medians = (
        persons.dropna(subset=["score", "region"])
        .groupby("region")
        .apply(lambda g: weighted_quantile(g["score"], 0.5, g["weight"]))
    )
    print("\nRegion weighted medians of score:")
    print(region_medians)
    persons["region_median_score"] = persons["region"].map(region_medians)
    persons["poor_rel_region"] = (
        persons["score"] > persons["region_median_score"]
    ).astype(int)

    h("4. HEADCOUNT / INTENSITY / MPI BY REGION")

    mpi_rows = []
    for r in REGION_ORDER:
        sub = persons[persons["region"] == r]
        H = weighted_rate(sub, "poor_af")
        poor_sub = sub[sub["poor_af"] == 1]
        if poor_sub.empty:
            A = float("nan")
        else:
            A = float(
                (poor_sub["c_score"] * poor_sub["weight"]).sum()
                / poor_sub["weight"].sum()
            )
        mpi_rows.append({"region": r, "H": H, "A": A, "MPI": H * A if A == A else np.nan})

    mpi_tbl = pd.DataFrame(mpi_rows)
    print(mpi_tbl.to_string(index=False))
    mpi_tbl.to_csv(OUT_DIR / "mpi_by_region.csv", index=False)

    h("5. DEPRIVATION RATES BY REGION")

    rate_rows = []
    for r in REGION_ORDER:
        sub = persons[persons["region"] == r]
        row = {"region": r}
        for col in indicators:
            row[col] = weighted_rate(sub.dropna(subset=[col]), col)
        rate_rows.append(row)
    rate_tbl = pd.DataFrame(rate_rows)
    print(rate_tbl.to_string(index=False))
    rate_tbl.to_csv(OUT_DIR / "deprivation_rates_by_region.csv", index=False)

    h("6. PCA SANITY CHECK")

    pca_df = persons[indicators].dropna()
    pca = PCA(n_components=min(4, len(indicators)))
    pca.fit(pca_df.to_numpy())
    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    pc1_loadings = pd.Series(pca.components_[0], index=indicators)
    print("PC1 loadings:")
    print(pc1_loadings.to_string())
    pc1_loadings.to_csv(OUT_DIR / "pca_pc1_loadings.csv", header=["loading"])

    h("7. WRITE FINAL PERSONS FRAME WITH INDICATORS")

    persons.to_parquet(OUT_DIR / "persons_indicators.parquet", index=False)
    print(f"\n[done] persons_indicators: {persons.shape}")


if __name__ == "__main__":
    main()
