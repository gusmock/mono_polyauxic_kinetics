# Native reproduction of the historical compute capsule

The scientific source, example data, metadata and licenses remain available here.
Use a separate Python 3.12 environment for this historical analysis. This capsule
is not the institutional ARCHÆA installation.

From this capsule directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
cd code
../.venv/bin/python demo.py
```

Place an authorized `data.xlsx` or `data.csv` in `code/` before running: the first
numeric column is time and the second is the response. If neither file exists,
the demonstration generates random synthetic data; those are not the original
study data. Outputs are written to `results/` beside `code/`.

The requirements describe the legacy Python dependencies and are not an approved
production environment. Record the versions used when reproducing results.
The published Code Ocean record retains its own immutable scientific provenance.
