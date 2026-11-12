from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from legacy_streamlit.analysis import run_analysis
from legacy_streamlit.core import gompertz_term_eq32, polyauxic_model

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
    with patch("legacy_streamlit.ui.run_analysis", wraps=run_analysis) as fit:
        at.button(key="run_analysis_button").click().run()
        assert not at.exception and not at.error
        assert fit.call_count == 1
        assert at.get("download_button")
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
