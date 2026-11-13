import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit", minversion="1.67.0")
from streamlit.testing.v1 import AppTest  # noqa: E402

from legacy_streamlit.analysis import run_analysis
from legacy_streamlit.attribution import PROFILE_LINKS
from legacy_streamlit.core import gompertz_term_eq32, polyauxic_model
from legacy_streamlit.i18n import LANGUAGES, TEXTS

ROOT = Path(__file__).resolve().parents[2]


def payload():
    t = np.linspace(0, 15, 25)
    y = polyauxic_model(t, [1.0, 12.0, 0.0, 2.0, 2.0], gompertz_term_eq32, 1)
    return pd.DataFrame({"t": t, "y": y}).to_csv(index=False).encode()


def app_with_data(language="English"):
    at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    at.selectbox(key="main_lang_selector").set_value(language)
    at.number_input(key="max_phases_input").set_value(1)
    at.get("file_uploader")[0].set_value(("synthetic.csv", payload(), "text/csv")).run()
    assert not at.exception and not at.error
    return at


@pytest.mark.parametrize("language", ["English", "Português (BR)", "Français (CA)"])
def test_anonymous_upload_fit_and_download_in_all_languages(language):
    at = app_with_data(language)
    lang = LANGUAGES[language]
    assert at.selectbox(key="csv_separator").value == "auto"
    assert at.selectbox(key="decimal_separator").value == "auto"
    assert at.selectbox(key="outlier_method_selector").value == "rout"
    text = "\n".join(m.value for m in at.markdown)
    assert "Prof. Dr. Gustavo Mockaitis" in text
    assert "Interdisciplinary Research Group of Biotechnology" in text
    assert "School of Agricultural Engineering, University of Campinas." in text
    assert "Campinas, SP, Brazil." in text
    assert all(url in text for _, url in PROFILE_LINKS)
    for doi in (
        "10.1007/s11538-026-01621-7",
        "10.48550/arXiv.2507.05960",
        "10.5281/zenodo.18025828",
        "10.24433/CO.0225069.v1",
    ):
        assert doi in text
    with patch("legacy_streamlit.ui.run_analysis", wraps=run_analysis) as fit:
        at.button(key="run_analysis_button").click().run()
        assert not at.exception and not at.error
        assert fit.call_count == 1
        assert at.get("download_button")
        assert_visible_parameters(at, lang)
        at.text_input(key="axis_x_custom").set_value("Time (d)").run()
        at.selectbox(key="plot_font_selector").set_value("DejaVu Sans").run()
        assert not at.exception and not at.error
        assert fit.call_count == 1
        assert at.session_state["analysis_result"]
        at.number_input(key="max_phases_input").set_value(2).run()
        assert "analysis_result" not in at.session_state
        assert fit.call_count == 1
    assert not any(b.key and ("login" in b.key or "logout" in b.key) for b in at.button)
    assert not any("email" in str(x.key) for x in at.text_input)


def assert_visible_parameters(at, lang):
    # A table nested in a collapsed expander is a regression, even if its data exist.
    assert all(not expander.dataframe for expander in at.expander)
    globals_ = [d.value for d in at.dataframe if TEXTS["table_col_param"][lang] in d.value]
    phases = [d.value for d in at.dataframe if f"p ± {TEXTS['table_col_se'][lang]}" in d.value]
    assert len(globals_) == len(phases) == 2
    bundle = at.session_state["analysis_result"]
    for global_table, phase_table, (name, family) in zip(
        globals_, phases, bundle["families"].items(), strict=True
    ):
        result = family["results"][at.selectbox(key=f"candidate_{name}").value]
        for i in (0, 1):
            value = global_table.iloc[i, 1]
            assert float(value.split(" ± ")[0]) == pytest.approx(result["theta"][i], rel=6e-6)
            if np.isfinite(result["se"][i]):
                assert float(value.split(" ± ")[1]) == pytest.approx(result["se"][i], rel=6e-6)
            else:
                assert TEXTS["not_estimable"][lang] in value
        n = result["n_phases"]
        assert len(phase_table) == n
        # Rows follow lag order, with a first-order phase kept first.
        indices = sorted(
            range(n),
            key=lambda j: (
                0 if result["use_first_order_phase1"] and j == 0 else 1,
                result["theta"][2 + 2 * n + j],
            ),
        )
        weights = np.exp(result["theta"][2 : 2 + n] - np.max(result["theta"][2 : 2 + n]))
        weights /= weights.sum()
        for row, j in enumerate(indices):
            p = phase_table.iloc[row, 1]
            assert float(p.split()[0]) == pytest.approx(weights[j], rel=6e-6)
            if n == 1:
                assert TEXTS["fixed_parameter"][lang] in p
            elif np.isfinite(result["se_p"][j]):
                assert float(p.split(" ± ")[1]) == pytest.approx(result["se_p"][j], rel=6e-6)
            for col, offset in ((2, 2 + n), (3, 2 + 2 * n)):
                cell = phase_table.iloc[row, col]
                if col == 3 and j == 0 and result["use_first_order_phase1"]:
                    assert cell == "N/A"
                    continue
                assert float(cell.split(" ± ")[0]) == pytest.approx(
                    result["theta"][offset + j], rel=6e-6
                )
                if np.isfinite(result["se"][offset + j]):
                    assert float(cell.split(" ± ")[1]) == pytest.approx(
                        result["se"][offset + j], rel=6e-6
                    )
                else:
                    assert TEXTS["not_estimable"][lang] in cell


def test_decimal_comma_upload_runs_without_manual_settings():
    source = pd.read_csv(io.BytesIO(payload()))
    content = source.to_csv(index=False, sep=";", decimal=",").encode()
    at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    at.number_input(key="max_phases_input").set_value(1)
    at.get("file_uploader")[0].set_value(("dados.csv", content, "text/csv")).run()
    assert not at.exception and not at.error
    at.button(key="run_analysis_button").click().run()
    assert not at.exception and not at.error
    bundle = at.session_state["analysis_result"]
    assert bundle["input_settings"] == {
        "delimiter": "auto",
        "decimal": "auto",
        "resolved_delimiter": ";",
        "resolved_decimal": ",",
    }
    assert_visible_parameters(at, "en")


def test_parameter_tables_follow_candidate_changes_without_recomputing():
    at = app_with_data()
    at.number_input(key="max_phases_input").set_value(2)
    with patch("legacy_streamlit.ui.run_analysis", wraps=run_analysis) as fit:
        at.button(key="run_analysis_button").click().run()
        assert not at.exception
        assert_visible_parameters(at, "en")
        for name in ("Gompertz", "Boltzmann"):
            results = at.session_state["analysis_result"]["families"][name]["results"]
            index = next(i for i, r in enumerate(results) if r["n_phases"] == 2)
            at.selectbox(key=f"candidate_{name}").set_value(index).run()
            assert not at.exception
            assert_visible_parameters(at, "en")
        assert fit.call_count == 1


def test_tiny_standard_errors_and_unavailable_errors_remain_distinct():
    from legacy_streamlit.plots import format_estimate

    assert format_estimate(0.0000123456, 0.0000000123456, "pt") == "1.23456e-05 ± 1.23456e-08"
    assert format_estimate(2, np.nan, "pt") == "2 ± não estimável"
    assert format_estimate(0, 0, "pt", fixed=True) == "0 (fixo)"
    assert format_estimate(2, 0, "pt") == "2 ± 0"


def test_upload_change_and_removal_discard_previous_results():
    at = app_with_data()
    at.button(key="run_analysis_button").click().run()
    assert at.session_state["analysis_result"]
    at.get("file_uploader")[0].set_value(("bad.csv", b"t,y\n0,1\n1,inf", "text/csv")).run()
    assert at.error and not at.exception
    assert "analysis_result" not in at.session_state
    at.get("file_uploader")[0].clear().run()
    assert not at.error and not at.exception
    assert "analysis_result" not in at.session_state


def test_sessions_do_not_share_analysis():
    first = app_with_data()
    first.button(key="run_analysis_button").click().run()
    second = AppTest.from_file(ROOT / "app.py").run()
    assert "analysis_result" not in second.session_state
    assert "analysis_result" in first.session_state


def test_malformed_math_axis_falls_back_without_losing_fit():
    at = app_with_data()
    at.button(key="run_analysis_button").click().run()
    at.text_input(key="axis_x_custom").set_value(r"$\notacommand$").run()
    assert at.warning and not at.exception
    assert "analysis_result" in at.session_state


def test_synthetic_example_can_replace_invalid_upload():
    at = app_with_data()
    at.get("file_uploader")[0].set_value(("bad.csv", b"t,y\n0,1\n1,inf", "text/csv")).run()
    assert at.error
    at.button(key="synthetic_example").click().run()
    assert not at.error and not at.exception
    at.get("file_uploader")[0].set_value(("bad2.csv", b"t,y\n0,1\n2,inf", "text/csv")).run()
    assert at.error and not at.exception
