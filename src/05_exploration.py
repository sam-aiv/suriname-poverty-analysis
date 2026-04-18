"""Step 5: exploration and visual audits for the appendix.

Produces the descriptive tables and figures used in Appendix 8.2:
    - weighted population totals and share by region
    - weighted age distribution (histogram + KDE) by region
    - deprivation rates by region (bar chart)
    - AF poverty headcount by region
    - poverty heatmap: score x region
    - missingness audits for the indicator inputs

Figures are saved under appendix_audits/figures/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    FIG_DIR,
    OUT_DIR,
    REGION_ORDER,
    REGION_PALETTE,
    h,
    missing_table,
    weighted_rate,
)

sns.set_theme(style="whitegrid", context="talk")


def weighted_population_table(persons: pd.DataFrame) -> pd.DataFrame:
    tab = (
        persons.dropna(subset=["region", "weight"])
        .groupby("region")["weight"]
        .sum()
        .reindex(REGION_ORDER)
        .rename("pop_est")
        .to_frame()
    )
    tab["share"] = tab["pop_est"] / tab["pop_est"].sum()
    return tab


def plot_age_distribution(persons: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in REGION_ORDER:
        sub = persons[(persons["region"] == r) & persons["q01_01"].notna()]
        if sub.empty:
            continue
        sns.kdeplot(
            sub["q01_01"],
            weights=sub["weight"],
            label=r,
            color=REGION_PALETTE[r],
            fill=True,
            alpha=0.25,
            ax=ax,
        )
    ax.set_xlabel("Age")
    ax.set_ylabel("Weighted density")
    ax.set_title("Weighted age distribution by region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_distribution_by_region.png", dpi=180)
    plt.close(fig)


def plot_deprivation_bars(persons: pd.DataFrame) -> None:
    cols = ["education_pov", "employment_pov", "electricity_pov", "econ_pov"]
    rows = []
    for r in REGION_ORDER:
        sub = persons[persons["region"] == r]
        for c in cols:
            rows.append(
                {"region": r, "indicator": c, "rate": weighted_rate(sub.dropna(subset=[c]), c)}
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="indicator",
        y="rate",
        hue="region",
        palette=REGION_PALETTE,
        hue_order=REGION_ORDER,
        ax=ax,
    )
    ax.set_ylabel("Weighted deprivation rate")
    ax.set_xlabel("")
    ax.set_title("Deprivation rates by region (SSLC 2022, weighted)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deprivation_rates_by_region.png", dpi=180)
    plt.close(fig)


def plot_score_heatmap(persons: pd.DataFrame) -> None:
    tab = (
        persons.dropna(subset=["score", "region", "weight"])
        .groupby(["region", "score"])["weight"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(REGION_ORDER)
    )
    # Row-normalise so each region sums to 1
    tab_norm = tab.div(tab.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        tab_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Weighted share"},
        ax=ax,
    )
    ax.set_xlabel("Deprivation score (0-4)")
    ax.set_ylabel("")
    ax.set_title("Distribution of deprivation scores by region")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_heatmap.png", dpi=180)
    plt.close(fig)


def plot_mpi_bars() -> None:
    mpi = pd.read_csv(OUT_DIR / "mpi_by_region.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=mpi,
        x="region",
        y="MPI",
        order=REGION_ORDER,
        palette=[REGION_PALETTE[r] for r in REGION_ORDER],
        ax=ax,
    )
    ax.set_title("Alkire-Foster MPI by region (K = 1/3)")
    ax.set_ylabel("MPI = H x A")
    ax.set_xlabel("")
    for i, row in mpi.iterrows():
        ax.text(i, row["MPI"], f"{row['MPI']:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mpi_by_region.png", dpi=180)
    plt.close(fig)


def main() -> None:
    h("1. LOAD PERSONS WITH INDICATORS")
    persons = pd.read_parquet(OUT_DIR / "persons_indicators.parquet")
    print(f"persons: {persons.shape}")

    h("2. WEIGHTED POPULATION TABLE")
    pop = weighted_population_table(persons)
    print(pop.to_string())
    pop.to_csv(OUT_DIR / "weighted_population_by_region.csv")

    h("3. MISSINGNESS AUDIT (INDICATOR INPUTS)")
    audit_cols = [
        "q01_01",
        "q01_03",
        "q04_05",
        "q05_02",
        "q13_17",
        "food_exp",
        "education_pov",
        "employment_pov",
        "electricity_pov",
        "econ_pov",
    ]
    miss = missing_table(persons, audit_cols)
    print(miss.to_string(index=False))
    miss.to_csv(OUT_DIR / "indicators_missingness.csv", index=False)

    h("4. FIGURES")
    plot_age_distribution(persons)
    plot_deprivation_bars(persons)
    plot_score_heatmap(persons)
    plot_mpi_bars()

    print(f"\n[done] figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
