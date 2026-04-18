"""Step 4: the survey-weighted logistic regressions (RQ1, RQ2, RQ3, combined).

Each research question uses a weighted binomial GLM with an HC1-robust
covariance estimator. RQ3 (electricity + region) is estimated both as a
standard GLM and as a ridge logistic fallback (C = 0.1) because the
interior × grid cell is near-separated.

Outputs (under appendix_exports/):
    rq1_tidy.csv, rq1_metrics.csv, rq1_ames.csv
    rq2_tidy.csv, rq2_metrics.csv, rq2_ames.csv
    rq3_tidy.csv, rq3_metrics.csv, rq3_ridge_coefs.csv
    combined_tidy.csv, combined_metrics.csv
    vif_report.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import (
    OUT_DIR,
    REGION_ORDER,
    ame_region_switch,
    h,
    tidy_glm_res,
    weighted_ame_binary,
    weighted_ame_continuous,
    weighted_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fit_glm(formula: str, df: pd.DataFrame, weight: str = "weight"):
    """Fit a weighted binomial GLM with HC1 robust SEs."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    w = df[weight].to_numpy()
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w)
    return model.fit(cov_type="HC1"), X, y


def vif_frame(X: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    cols = [c for c in X.columns if c.lower() != "intercept"]
    X_ = X[cols].to_numpy()
    for i, c in enumerate(cols):
        try:
            rows.append({"model": label, "term": c, "vif": variance_inflation_factor(X_, i)})
        except Exception:  # noqa: BLE001
            rows.append({"model": label, "term": c, "vif": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1 — multidimensional poverty ~ region + age + gender
# ---------------------------------------------------------------------------


def rq1(persons: pd.DataFrame) -> pd.DataFrame:
    h("RQ1: multidimensional poverty ~ region + age + gender")
    df = persons.dropna(subset=["poor_rel_region", "region", "q01_01", "is_female", "weight"]).copy()
    df["age"] = df["q01_01"].astype(float)

    formula = "poor_rel_region ~ C(region, Treatment('Great Paramaribo')) + age + is_female"
    res, X, y = fit_glm(formula, df)
    print(res.summary())

    tidy = tidy_glm_res(res)
    tidy.to_csv(OUT_DIR / "rq1_tidy.csv")

    p_hat = res.predict(X)
    acc, prec, rec, cm = weighted_metrics(y.iloc[:, 0].to_numpy(), p_hat.to_numpy(), df["weight"])
    pd.DataFrame(
        [{"accuracy": acc, "precision": prec, "recall": rec}]
    ).to_csv(OUT_DIR / "rq1_metrics.csv", index=False)
    cm.to_csv(OUT_DIR / "rq1_confusion.csv")

    ames = {
        "age": weighted_ame_continuous(res, df, "age", h=1.0),
        "is_female": weighted_ame_binary(res, df, "is_female"),
        "region_RoC": ame_region_switch(res, df, "Great Paramaribo", "Rest of Coast"),
        "region_Interior": ame_region_switch(res, df, "Great Paramaribo", "Interior"),
    }
    pd.Series(ames, name="ame").to_csv(OUT_DIR / "rq1_ames.csv")

    return vif_frame(X, "rq1")


# ---------------------------------------------------------------------------
# RQ2 — employment_pov ~ region × ... (working age 15-64)
# ---------------------------------------------------------------------------


def rq2(persons: pd.DataFrame) -> pd.DataFrame:
    h("RQ2: employment deprivation ~ region + age + gender (15-64)")
    df = persons.copy()
    df = df[df["working_age"] == 1]
    df = df.dropna(subset=["employment_pov", "region", "q01_01", "is_female", "weight"])
    df["age"] = df["q01_01"].astype(float)

    formula = "employment_pov ~ C(region, Treatment('Great Paramaribo')) + age + is_female"
    res, X, y = fit_glm(formula, df)
    print(res.summary())

    tidy_glm_res(res).to_csv(OUT_DIR / "rq2_tidy.csv")
    p_hat = res.predict(X)
    acc, prec, rec, cm = weighted_metrics(y.iloc[:, 0].to_numpy(), p_hat.to_numpy(), df["weight"])
    pd.DataFrame(
        [{"accuracy": acc, "precision": prec, "recall": rec}]
    ).to_csv(OUT_DIR / "rq2_metrics.csv", index=False)
    cm.to_csv(OUT_DIR / "rq2_confusion.csv")

    ames = {
        "age": weighted_ame_continuous(res, df, "age", h=1.0),
        "is_female": weighted_ame_binary(res, df, "is_female"),
        "region_RoC": ame_region_switch(res, df, "Great Paramaribo", "Rest of Coast"),
        "region_Interior": ame_region_switch(res, df, "Great Paramaribo", "Interior"),
    }
    pd.Series(ames, name="ame").to_csv(OUT_DIR / "rq2_ames.csv")

    return vif_frame(X, "rq2")


# ---------------------------------------------------------------------------
# RQ3 — electricity_pov ~ region + age + gender, 25-64, with ridge fallback
# ---------------------------------------------------------------------------


def rq3(persons: pd.DataFrame) -> pd.DataFrame:
    h("RQ3: electricity deprivation ~ region + age + gender (25-64)")
    df = persons.copy()
    df = df[df["q01_01"].between(25, 64)]
    df = df.dropna(subset=["electricity_pov", "region", "q01_01", "is_female", "weight"])
    df["age"] = df["q01_01"].astype(float)

    formula = "electricity_pov ~ C(region, Treatment('Great Paramaribo')) + age + is_female"

    try:
        res, X, y = fit_glm(formula, df)
        print(res.summary())
        tidy_glm_res(res).to_csv(OUT_DIR / "rq3_tidy.csv")
        p_hat = res.predict(X).to_numpy()
    except Exception as err:  # noqa: BLE001
        print(f"[warn] GLM failed ({err}); continuing with ridge only")
        y, X = patsy.dmatrices(formula, df, return_type="dataframe")
        p_hat = None

    # Ridge fallback for quasi-separation (Interior x grid is small).
    cols = [c for c in X.columns if c.lower() != "intercept"]
    X_np = X[cols].to_numpy()
    y_np = y.iloc[:, 0].to_numpy()
    ridge = LogisticRegression(
        penalty="l2", C=0.1, solver="lbfgs", max_iter=5000
    ).fit(X_np, y_np, sample_weight=df["weight"].to_numpy())
    ridge_coefs = pd.Series(ridge.coef_[0], index=cols)
    print("Ridge coefficients (C=0.1):")
    print(ridge_coefs.to_string())
    ridge_coefs.to_csv(OUT_DIR / "rq3_ridge_coefs.csv", header=["coef"])

    if p_hat is None:
        p_hat = ridge.predict_proba(X_np)[:, 1]

    acc, prec, rec, cm = weighted_metrics(y_np, p_hat, df["weight"])
    pd.DataFrame(
        [{"accuracy": acc, "precision": prec, "recall": rec}]
    ).to_csv(OUT_DIR / "rq3_metrics.csv", index=False)
    cm.to_csv(OUT_DIR / "rq3_confusion.csv")

    return vif_frame(X, "rq3")


# ---------------------------------------------------------------------------
# Combined model — poor_rel_region ~ region + age + gender + employment + electricity
# ---------------------------------------------------------------------------


def combined(persons: pd.DataFrame) -> pd.DataFrame:
    h("COMBINED: poor_rel_region ~ region + age + gender + employment_pov + electricity_pov (25-64)")
    df = persons.copy()
    df = df[df["q01_01"].between(25, 64)]
    df = df.dropna(
        subset=[
            "poor_rel_region",
            "region",
            "q01_01",
            "is_female",
            "employment_pov",
            "electricity_pov",
            "weight",
        ]
    )
    df["age"] = df["q01_01"].astype(float)

    formula = (
        "poor_rel_region ~ C(region, Treatment('Great Paramaribo')) "
        "+ age + is_female + employment_pov + electricity_pov"
    )
    res, X, y = fit_glm(formula, df)
    print(res.summary())

    tidy_glm_res(res).to_csv(OUT_DIR / "combined_tidy.csv")
    p_hat = res.predict(X)
    acc, prec, rec, cm = weighted_metrics(y.iloc[:, 0].to_numpy(), p_hat.to_numpy(), df["weight"])
    pd.DataFrame(
        [{"accuracy": acc, "precision": prec, "recall": rec}]
    ).to_csv(OUT_DIR / "combined_metrics.csv", index=False)
    cm.to_csv(OUT_DIR / "combined_confusion.csv")

    return vif_frame(X, "combined")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    persons = pd.read_parquet(OUT_DIR / "persons_indicators.parquet")
    print(f"persons: {persons.shape}")
    print(f"regions present: {sorted(persons['region'].dropna().unique())}")
    print(f"expected region order: {REGION_ORDER}")

    vifs = [rq1(persons), rq2(persons), rq3(persons), combined(persons)]
    all_vif = pd.concat(vifs, ignore_index=True)
    all_vif.to_csv(OUT_DIR / "vif_report.csv", index=False)
    h("VIF REPORT")
    print(all_vif.to_string(index=False))
    print(f"\n[done] results written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
