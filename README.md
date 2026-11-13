# Polyauxic Modeling Platform — Streamlit

Open-access fitting of increasing mono- and polyauxic responses with the
modified Gompertz and Boltzmann equations. No registration or login is required.

Scientific reference: Gustavo Mockaitis,
[Bulletin of Mathematical Biology](https://doi.org/10.1007/s11538-026-01621-7).
The archived [Code Ocean capsule](https://doi.org/10.24433/CO.0225069.v1) and
[Zenodo release](https://doi.org/10.5281/zenodo.18025828) remain historical
reproduction references; this maintenance build contains documented corrections.

## Install and run

Python **3.12 or later** is required by the pinned numerical dependencies.
Python 3.12 is the tested version. **`git pull` updates source only: also install
the new requirements and restart the service. Python 3.9 environments must be
recreated with Python 3.12.** Use a dedicated native virtual environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_legacy_frontend.py
.venv/bin/python -m streamlit run app.py
```

The security verifier checks JavaScript actually installed in the Streamlit
wheel, including DOMPurify ≥3.4.17 and Moment ≥2.31.0. An institutional external
rescan remains necessary after deployment. See the
[maintenance and deployment notes](docs/LEGACY_STREAMLIT_MAINTENANCE.md).

## Data and results

Upload UTF-8 CSV or XLSX with a header row and 1–5 pairs of time/response
columns. Column separators (comma, semicolon or tab) and decimal separators
(dot or comma) are detected automatically. Manual overrides remain available.
The interface shows the interpreted format; mixed decimal conventions require
an explicit choice or correction of the file. Thousands separators are unsupported.
Limits: 5 MB, 5,000 rows and 10,000 paired observations. Blank pairs are allowed;
partial pairs, nonnumeric/infinite values, negative times and formulas are
rejected with an explanation. Keep physical units consistent across replicates.

The synthetic example is available without a file. Set the maximum number of
phases and run the analysis. Figures support configurable fonts, axis labels
and SVG downloads. Parameter estimates and their standard errors are visible
directly below each candidate's plot, with six significant digits. Fixed and
unestimable parameters are labelled explicitly. Excel exports include numeric parameters and standard
errors, original observations, residuals, exclusions, all candidates, criteria,
seed, input hash and build identifier. English, Portuguese and French are supported.

Data and results stay in session memory. Download them before closing the page.
Progress and completed fits appear during computation. The elapsed analysis time
is recorded in the interface and export. Changing display settings does not repeat fitting; changing data or scientific
settings invalidates previous results. Two analyses can execute concurrently.
Each optimization has a 120-second budget. The complete analysis is no longer
cut off after 180 seconds; callers may explicitly request an overall time budget.
Progress remains visible while the requested candidates are processed.

## Scientific scope

The equations describe **increasing** responses. A zero final asymptote is no
longer offered as a substitute for a decreasing-response model. A first-order
phase is considered when the fitted first lag is approximately zero.

The original robust/FDR outlier option is again the default, with Q = 1%.
Users can explicitly select no removal or simple MAD. The robust/FDR procedure
retains the legacy Charbonnier + MAD + Benjamini–Hochberg approach; it is labelled
as a legacy approximation, not as a newly validated implementation of ROUT.
Candidates retaining different observations are not ranked automatically.

Selection preserves the first-local-minimum policy within each family.
The search again uses the original full coordinates, population, bounds and
DE/L-BFGS-B strategy. A failed refinement near coincident lag times can use
SLSQP with explicit chronological constraints; the diagnostic records which
refinement succeeded. The redundant reference logit is removed after fitting
when estimating covariance, rather than changing the stochastic search.
AIC/AICc/BIC use the independent fitted parameters plus one residual-variance
parameter. Fixed asymptotes, the reference logit and the unused first-order lag
are not counted as estimated mean-model parameters. Standard errors use a
residual Jacobian and are withheld for rank deficiency or failed convergence.
When a constraint is active and the covariance is computable, standard errors
remain visible with a warning that the local linear approximation ignores the
constraint. They are not validated confidence intervals. Constraints, conditioning
and optimizer diagnostics are included in the Excel export.

## Author

**Prof. Dr. Gustavo Mockaitis — GBMA / FEAGRI / UNICAMP.**
Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and
Environment, School of Agricultural Engineering, University of Campinas.
Campinas, SP, Brazil. [ORCID](https://orcid.org/0000-0002-4231-1056) ·
[Lattes](http://lattes.cnpq.br/1400402042483439) ·
[Institutional profile](http://feagri.unicamp.br/mockaitis).
The application also preserves the author's photo, professional profiles and
full article, preprint, Zenodo and Code Ocean references.

## Code and tests

`app.py` is the entry point. `legacy_streamlit/` separates data validation,
scientific core, orchestration, plots, exports, translations and UI.
`polyauxic_core.py` remains a compatibility import for the public core.
Historical `1.1.0/` and `Codeocean_capsule_V1.0.0/` are not runtime dependencies.

```bash
.venv/bin/python -m pip install pytest==9.1.1 ruff==0.16.3
.venv/bin/python -m pytest -c tests/legacy_streamlit/pytest.ini tests/legacy_streamlit
.venv/bin/python -m ruff check --config legacy_streamlit/ruff.toml app.py legacy_streamlit tests/legacy_streamlit scripts/check_legacy_frontend.py
```
