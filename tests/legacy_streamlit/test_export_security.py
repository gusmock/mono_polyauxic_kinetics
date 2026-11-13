import importlib.util
import io
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from legacy_streamlit.analysis import AnalysisConfig, run_analysis
from legacy_streamlit.core import gompertz_term_eq32, polyauxic_model
from legacy_streamlit.report import generate_excel_report

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "frontend_check", ROOT / "scripts/check_legacy_frontend.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_export_is_numeric_auditable_and_does_not_create_formulas():
    t = np.linspace(0, 15, 25)
    y = polyauxic_model(t, [1.0, 12.0, 0.0, 2.0, 2.0], gompertz_term_eq32, 1)
    analysis = run_analysis(t, y, AnalysisConfig(max_phases=1))
    best = {name: f["selected"] for name, f in analysis["families"].items()}
    result = generate_excel_report(
        best,
        [{"t": t, "y": y, "name": "=2+2"}],
        ("y_i", "y_f"),
        "r_max",
        "en",
        analysis=analysis,
        metadata={"x_axis": '=HYPERLINK("https://example.com")'},
    )
    book = load_workbook(io.BytesIO(result), data_only=False)
    assert {"Data", "Audit", "Candidates"} <= set(book.sheetnames)
    assert any(s.endswith("_Parameters") for s in book.sheetnames)
    for sheet in book:
        for row in sheet:
            for cell in row:
                assert cell.data_type != "f"
                assert cell.hyperlink is None
    data = book["Gom_1_S_Data"]
    assert data.max_row == 26 and data["E2"].data_type == "n"
    assert book["Gom_1_S_Parameters"]["C2"].data_type == "n"
    candidate_headers = [c.value for c in book["Candidates"][1]]
    assert {"uncertainty_status", "active_constraints", "optimizer_refinement"} <= set(
        candidate_headers
    )


def test_security_checker_detects_old_and_new_embedded_versions(tmp_path):
    (tmp_path / "a.js").write_text("x.version=`3.4.17`,x.removed=[]; x.sanitize=fn; // DOMPurify")
    (tmp_path / "b.js").write_text('m.version="2.31.0",a(j),m.fn=Y; // moment.js')
    findings, unknown = check.inspect_assets(tmp_path)
    assert not unknown
    assert findings["DOMPurify"] == [("a.js", "3.4.17")]
    (tmp_path / "old.js").write_text('p.version="3.2.6",p.removed=[]; p.sanitize=fn; // DOMPurify')
    findings, _ = check.inspect_assets(tmp_path)
    assert any(
        tuple(map(int, v.split("."))) < check.MINIMUM["DOMPurify"] for _, v in findings["DOMPurify"]
    )


def test_security_checker_unknown_implementation_is_not_passed(tmp_path):
    (tmp_path / "purify.js").write_text("/* DOMPurify */ p.sanitize = fn;")
    _, unknown = check.inspect_assets(tmp_path)
    assert unknown
