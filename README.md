# Multidimensional Poverty and the Labour Market: A Regional Analysis of Suriname

Reproducible Python pipeline for my MSc Applied Data Science thesis
(Utrecht University, July 2025). The study uses the 2022 Suriname Survey
of Living Conditions (SSLC, N = 7,715) to estimate how **region**,
**employment**, and **electricity access** relate to multidimensional
poverty, using survey-weighted logistic regression and an
Alkire–Foster–style MPI as a robustness check.

**Author:** Samantha Aivazi
**Supervisor:** Dr. Yolanda Grift
**Second reader:** Dr. Rosita Sobhie

---

## What this repo contains

```
suriname-poverty-analysis/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md                  # how to obtain the SSLC 2022 microdata
└── src/
    ├── 01_data_loading.py         # import RT001 / RT002 / RT140, schema checks
    ├── 02_data_cleaning.py        # keys, merges, region/sex recodes
    ├── 03_indicators.py           # 4 binary deprivations + composite scores
    ├── 04_regressions.py          # RQ1–RQ3 + combined model (GLM + ridge)
    ├── 05_exploration.py          # descriptive figures and audit visuals
    └── utils.py                   # shared helpers (weighted stats, AMEs)
```

## Research questions

- **RQ1** — To what extent does geographic region predict relative
  (within-region) poverty after controlling for age and gender?
- **RQ2** — Do the effects of employment deprivation on relative poverty
  differ across regions (region × employment interactions)?
- **RQ3** — What is the association between electricity deprivation and
  relative poverty, controlling for region and demographics, among
  adults of working age?
- **Combined** — When region, employment and electricity are modelled
  jointly, which factors dominate?

## Headline findings

| Model    | Sample        | Accuracy | AUC   |
|----------|---------------|---------:|------:|
| RQ1      | All ages      |  0.606   | 0.609 |
| RQ2      | Ages 15–64    |  0.889   | 0.900 |
| RQ3      | Ages 25–64    |  0.630   | 0.688 |
| Combined | Ages 25–64    |  0.855   | 0.850 |

Weighted average marginal effects from the combined model: being
employed reduces P(poor) by ~0.82; gaining electricity access reduces
P(poor) by ~0.58. The Alkire–Foster–style MPI places the Interior at
H ≈ 0.82, A ≈ 0.65, **MPI ≈ 0.53**, versus 0.056 in Great Paramaribo
and 0.080 in the Rest of Coast.

## Data

This project uses three publicly-available files from the **Inter-American
Development Bank's 2022 Suriname Survey of Living Conditions (SSLC)**:

| File | Rows × cols | Description |
|---|---|---|
| `02-databases-rt002-weighted-public.dta` | 7,715 × 594 | Persons |
| `02-databases-rt001-weighted-public.dta` | 2,542 × 90 | Housing |
| `02-databases-rt140-weighted-public.dta` | 88,375 × 20 | Food transactions |

The data files are **not** included in this repository. See
[`data/README.md`](data/README.md) for download instructions.

## How to reproduce

```bash
# 1. clone and install
git clone https://github.com/<you>/suriname-poverty-analysis.git
cd suriname-poverty-analysis
python -m venv .venv && source .venv/bin/activate     # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt

# 2. place the three .dta files under data/
#    (see data/README.md for download instructions)

# 3. run the pipeline in order
python src/01_data_loading.py
python src/02_data_cleaning.py
python src/03_indicators.py
python src/04_regressions.py
python src/05_exploration.py
```

Intermediate CSVs are written to `appendix_exports/` and figures to
`appendix_audits/figures/`.

## Method in one paragraph

Each person is scored on four binary deprivations — education
(age-bounded enrolment/attendance), employment (ages 15–64 from the 7-day
reference question), electricity (grid access from the housing module),
and an economic dimension (household per-capita annualised food
expenditure below the weighted 20th percentile ≈ SRD 8,475.60).
Summing these gives a 0–4 score, and the dependent variable
`poor_rel_region` flags individuals whose score strictly exceeds their
region's person-weighted median. Survey-weighted binomial GLMs with HC1
standard errors are fitted for RQ1–RQ3 and a combined specification;
ridge-penalised logistic regression is used as a fallback where
quasi-separation is detected (notably RQ3). Weighted average marginal
effects are computed manually via counterfactual predictions so that
survey weights apply.

## License

Code is released under the MIT License (see `LICENSE`). The SSLC data
itself is owned and distributed by the Inter-American Development Bank
under its own terms — please consult the IDB before redistributing it.

## Citation

If you use this code, please cite the thesis:

> Aivazi, S. (2025). *Multidimensional Poverty and the Labour Market:
> A Regional Analysis of Suriname.* MSc thesis, Applied Data Science,
> Utrecht University.
