# Data

This project uses the **2022 Suriname Survey of Living Conditions (SSLC)**,
a nationally representative household survey conducted by the
Inter-American Development Bank (IDB) between January and December 2022.
It covers approximately 2,500 households and 7,500 individuals,
stratified across three domains: Great Paramaribo (urban),
Rest of the Coastal Region (semi-urban), and the Interior (rural).

## Files expected in this folder

Three Stata files from the SSLC public release:

| File | Shape | Contents |
|---|---|---|
| `02-databases-rt002-weighted-public.dta` | 7,715 × 594 | Persons — demographics, education, employment |
| `02-databases-rt001-weighted-public.dta` | 2,542 × 90  | Housing — dwelling characteristics, infrastructure |
| `02-databases-rt140-weighted-public.dta` | 88,375 × 20 | Food — household food acquisition records |

The scripts in `src/` expect these files at `data/<filename>.dta`. Update
the paths in `src/utils.py` (variable `DATA_DIR`) if you store them
elsewhere.

## Where to get them

The SSLC 2022 microdata is distributed by the IDB. Search the IDB
Numbers for Development (Números para el Desarrollo) microdata portal
for the *Suriname Survey of Living Conditions 2022*:

- https://mydata.iadb.org/
- https://publications.iadb.org/

Access is free but typically requires a (free) account and acceptance of
the IDB data-use terms. Once downloaded, place the three `.dta` files
directly in this `data/` folder.

## Not redistributed here

The files are not committed to this repository because:

1. Redistribution rights sit with the IDB, not the author of this thesis.
2. The combined archive is several hundred MB.

`.gitignore` is configured to keep `.dta` files out of git while
preserving this README.

## Ethical and legal notes

- All analyses use anonymised microdata; no personally identifiable
  information is retained.
- Results are reported as regional aggregates.
- See Section 3.4 of the thesis for the full ethical and legal
  discussion.
