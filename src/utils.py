"""Shared helpers for the SSLC 2022 poverty pipeline.

Centralises file paths, survey-weighted statistics, and the weighted
average marginal effect (AME) routines used by the regression scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Update this if your data lives elsewhere.
DATA_DIR = Path(os.environ.get("SSLC_DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("SSLC_OUT_DIR", "appendix_exports"))
FIG_DIR = Path(os.environ.get("SSLC_FIG_DIR", "appendix_audits/figures"))

PERSONS_PATH = DATA_DIR / "02-databases-rt002-weighted-public.dta"
HOUSING_PATH = DATA_DIR / "02-databases-rt001-weighted-public.dta"
FOOD_PATH = DATA_DIR / "02-databases-rt140-weighted-public.dta"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Coding maps
# ---------------------------------------------------------------------------

REGION_MAP = {1: "Great Paramaribo", 2: "Rest of Coast", 3: "Interior"}
REGION_ORDER = ["Great Paramaribo", "Rest of Coast", "Interior"]
REGION_PALETTE = {
    "Great Paramaribo": "#377eb8",
    "Rest of Coast": "#ff7f00",
    "Interior": "#4daf4a",
}

# q13_17 (main lighting source) — 1 is grid, everything else is deprived in the baseline.
ELECTRIC_GRID = [1]
ELECTRIC_OTHER = [2, 3, 4, 5, 6, 7, 8]  # solar (6) flipped in the sensitivity alt

# ---------------------------------------------------------------------------
# Utility printing
# ---------------------------------------------------------------------------


def h(title: str) -> None:
    """Print a simple section header."""
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def short_info(df: pd.DataFrame, name: str, show_head: bool = True) -> None:
    """Dump shape, head dtypes, and (optionally) the first 3 rows."""
    print(f"\n[{name}] shape: {df.shape}")
    print(df.dtypes.head(12))
    if show_head:
        print(f"\n[{name}] head(3):")
        print(df.head(3))

    buf = [f"{name} shape: {df.shape}\n", df.dtypes.to_string()]
    path = OUT_DIR / f"{name.lower()}_schema.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))


def must_have(df: pd.DataFrame, cols: list[str], name: str) -> bool:
    """Return True iff every `col` in `cols` exists in df."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[!] {name} missing columns: {missing}")
        return False
    print(f"[ok] {name} contains required columns: {cols}")
    return True


def missing_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        rows.append(
            {
                "column": c,
                "n_missing": int(df[c].isna().sum()) if c in df.columns else np.nan,
                "pct_missing": df[c].isna().mean() * 100 if c in df.columns else np.nan,
                "n_unique": df[c].nunique(dropna=True) if c in df.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


def weighted_quantile(values, q: float, w) -> float:
    """Linear-interpolation weighted quantile. Ignores NaNs in `values` and `w`."""
    values = np.asarray(values, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(w)
    values, w = values[mask], w[mask]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values, w = values[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, values))


def weighted_rate(df: pd.DataFrame, col: str, w: str = "weight") -> float:
    """Weighted mean of a binary column, dropping NaN pairs."""
    sub = df[[col, w]].dropna()
    if sub.empty:
        return float("nan")
    return float((sub[col].astype(float) * sub[w]).sum() / sub[w].sum())


def weighted_mean(x, w) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    if not mask.any():
        return float("nan")
    return float((x[mask] * w[mask]).sum() / w[mask].sum())


# ---------------------------------------------------------------------------
# Weighted AMEs via counterfactual predictions
# ---------------------------------------------------------------------------


def weighted_ame_binary(res, df: pd.DataFrame, var: str, weight: str = "weight", condition=None) -> float:
    """Weighted average change in predicted P(y=1) when a binary var goes 0 -> 1."""
    base_df = df if condition is None else df.loc[condition(df)].copy()
    if base_df.empty:
        return float("nan")
    d0 = base_df.copy()
    d1 = base_df.copy()
    d0[var] = 0
    d1[var] = 1
    p0 = res.predict(d0)
    p1 = res.predict(d1)
    return float(np.average((p1 - p0).to_numpy(), weights=base_df[weight].to_numpy()))


def weighted_ame_continuous(res, df: pd.DataFrame, var: str, h: float = 1.0, weight: str = "weight") -> float:
    """Weighted average derivative via finite difference of size h."""
    base_df = df.copy()
    d0 = base_df.copy()
    d1 = base_df.copy()
    d0[var] = d0[var].astype(float)
    d1[var] = d0[var] + h
    p0 = res.predict(d0)
    p1 = res.predict(d1)
    return float(np.average(((p1 - p0) / h).to_numpy(), weights=base_df[weight].to_numpy()))


def ame_region_switch(res, df: pd.DataFrame, from_region: str, to_region: str, weight: str = "weight") -> float:
    """AME of switching the `region` category column from one label to another."""
    d0 = df.copy()
    d1 = df.copy()
    d0["region"] = from_region
    d1["region"] = to_region
    p0 = res.predict(d0)
    p1 = res.predict(d1)
    return float(np.average((p1 - p0).to_numpy(), weights=df[weight].to_numpy()))


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def tidy_glm_res(res) -> pd.DataFrame:
    """Stable tidy table of coef / OR / CI / p-value for a statsmodels GLM result."""
    idx = res.params.index
    ci = res.conf_int()
    ci_low = ci.iloc[:, 0]
    ci_high = ci.iloc[:, 1]
    out = pd.DataFrame(index=idx)
    out["coef"] = pd.Series(res.params, index=idx)
    out["ci_low_logit"] = pd.Series(ci_low, index=idx)
    out["ci_high_logit"] = pd.Series(ci_high, index=idx)
    out["p_value"] = pd.Series(res.pvalues, index=idx)
    out["odds_ratio"] = np.exp(out["coef"])
    out["or_ci_low"] = np.exp(out["ci_low_logit"])
    out["or_ci_high"] = np.exp(out["ci_high_logit"])
    return out[
        [
            "coef",
            "odds_ratio",
            "ci_low_logit",
            "ci_high_logit",
            "or_ci_low",
            "or_ci_high",
            "p_value",
        ]
    ]


def weighted_metrics(y_true, y_prob, w, thr: float = 0.5):
    """Weighted accuracy / precision / recall + weighted confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    y_hat = (np.asarray(y_prob) >= thr).astype(int)
    acc = accuracy_score(y_true, y_hat, sample_weight=w)
    prec = precision_score(y_true, y_hat, sample_weight=w, zero_division=0)
    rec = recall_score(y_true, y_hat, sample_weight=w)
    cm = confusion_matrix(y_true, y_hat, sample_weight=w, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
    )
    return acc, prec, rec, cm_df
