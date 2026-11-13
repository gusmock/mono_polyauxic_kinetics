"""Anonymous Streamlit UI. Numerical results are cached only in the user's session."""

import hashlib
import logging
import secrets
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.mathtext import MathTextParser

from . import BUILD
from .analysis import MODELS, AnalysisBusy, AnalysisConfig, analysis_key, run_analysis
from .attribution import render_attribution
from .core import gompertz_term_eq32, polyauxic_model
from .data import InputDataError, read_upload
from .i18n import LANGUAGES, TEXTS
from .labels import boundary_parameter_labels, get_full_model_label, normalize_axis_math_label
from .plots import (
    PLOT_FONT_OPTIONS,
    PLOT_LOCK,
    display_single_fit,
    format_estimate,
    plot_final_summary,
    plot_metrics_summary,
    plot_raw_data,
)
from .report import generate_excel_report

LOGGER = logging.getLogger(__name__)


def _example():
    t = np.linspace(0, 15, 25)
    y = polyauxic_model(t, [1.0, 12.0, 0.0, 2.0, 2.0], gompertz_term_eq32, 1)
    return pd.DataFrame({"time": t, "response": y}).to_csv(index=False).encode()


def _axis(text, fallback, lang):
    label = normalize_axis_math_label(text) if text.strip() else fallback
    try:
        MathTextParser("agg").parse(label)
    except ValueError:
        st.warning(TEXTS["bad_axis"][lang])
        return fallback
    return label


def _render_results(bundle, replicates, lang, x_label, y_label, content_hash):
    best = {name: family["selected"] for name, family in bundle["families"].items()}
    tabs = st.tabs([TEXTS["tab_gompertz"][lang], TEXTS["tab_boltzmann"][lang]])
    for tab, (name, family) in zip(tabs, bundle["families"].items(), strict=False):
        with tab:
            if family["warning"]:
                st.warning(TEXTS[family["warning"]][lang])
            if family["results"]:
                selected_index = next(
                    (i for i, r in enumerate(family["results"]) if r is family["selected"]), 0
                )
                candidate_index = st.selectbox(
                    TEXTS["candidate"][lang],
                    list(range(len(family["results"]))),
                    index=selected_index,
                    format_func=lambda i, family=family, name=name: (
                        f"{get_full_model_label(family['results'][i], name, lang)} · {family['results'][i]['n_phases']}"
                    ),
                    key=f"candidate_{name}",
                )
                result = family["results"][candidate_index]
                if result.get("uncertainty_status") == "approximate_at_boundary":
                    st.warning(
                        TEXTS["uncertainty_boundary"][lang].format(
                            ", ".join(boundary_parameter_labels(result))
                        )
                    )
                elif not result["uncertainty_reliable"]:
                    st.warning(TEXTS["uncertainty"][lang])
                if not result["converged"]:
                    st.warning(TEXTS["no_convergence"][lang])
                with PLOT_LOCK:
                    display_single_fit(
                        result,
                        replicates,
                        name,
                        MODELS[name],
                        "tab:blue" if name == "Gompertz" else "tab:orange",
                        x_label,
                        y_label,
                        ("y_i", "y_f"),
                        "r_max",
                        lang,
                        force_yi=bundle["config"]["force_yi"],
                    )
            if family["results"]:
                rows = []
                for r in family["results"]:
                    rows.append(
                        {
                            "F": r["n_phases"],
                            TEXTS["full_model_col"][lang]: get_full_model_label(r, name, lang),
                            **r["metrics"],
                            "converged": r["converged"],
                            "selected": r is family["selected"],
                            "best_for_phase": any(
                                r is candidate for candidate in family["by_phase"]
                            ),
                        }
                    )
                summary = pd.DataFrame(rows)
                st.subheader(TEXTS["table_title"][lang])
                st.dataframe(
                    summary.style.apply(
                        lambda row: (
                            [
                                "background-color: #d7e8ff; color: #111; font-weight: bold"
                                if row["selected"]
                                else (
                                    "background-color: #d4edda; color: #111"
                                    if row["best_for_phase"]
                                    else ""
                                )
                            ]
                            * len(row)
                        ),
                        axis=1,
                    ),
                    hide_index=True,
                )
                st.caption(TEXTS["legend_final_best"][lang])
                st.caption(TEXTS["legend_phase_best"][lang])
            if family["selected"]:
                st.info(TEXTS["selection_logic_body"][lang].format(family["criterion"]))
                st.success(
                    TEXTS["best_model_msg"][lang].format(
                        family["selected"]["n_phases"], family["criterion"]
                    )
                )
                with PLOT_LOCK:
                    plot_metrics_summary(family["by_phase"], name, lang)
    if any(r is not None for r in best.values()):
        with PLOT_LOCK:
            plot_final_summary(replicates, best, lang, x_label, y_label)
    for name, n, key in bundle["warnings"]:
        st.caption(f"{name}, {n}: {TEXTS[key][lang]}")
    report = generate_excel_report(
        best,
        replicates,
        ("y_i", "y_f"),
        "r_max",
        lang,
        analysis=bundle,
        metadata={
            "build": BUILD,
            "sha256": content_hash,
            "x_axis": x_label,
            "y_axis": y_label,
            "selection": "first_local_minimum",
        },
    )
    st.download_button(
        TEXTS["download_excel"][lang],
        report,
        "polyauxic_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results",
        on_click="ignore",
    )


def _clear_example():
    st.session_state.pop("example_active", None)


def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Modeling Platform")
    language = st.sidebar.selectbox(
        "Language / Idioma / Langue", list(LANGUAGES), key="main_lang_selector"
    )
    lang = LANGUAGES[language]
    st.title(TEXTS["app_title"][lang])
    st.caption(TEXTS["privacy"][lang])
    st.sidebar.header(TEXTS["sidebar_config"][lang])
    uploaded = st.sidebar.file_uploader(
        TEXTS["upload_label"][lang],
        type=["csv", "xlsx"],
        key="data_file_uploader",
        on_change=_clear_example,
    )
    st.sidebar.caption(TEXTS["limits"][lang])
    if st.sidebar.button(TEXTS["example"][lang], key="synthetic_example"):
        st.session_state["example_active"] = True
    delimiter = st.sidebar.selectbox(
        TEXTS["separator"][lang],
        ["auto", ";", ",", "\t"],
        format_func=lambda s: (
            TEXTS["automatic"][lang] if s == "auto" else ("Tab" if s == "\t" else s)
        ),
        key="csv_separator",
    )
    decimal = st.sidebar.selectbox(
        TEXTS["decimal"][lang],
        ["auto", ".", ","],
        format_func=lambda s: TEXTS["automatic"][lang] if s == "auto" else s,
        key="decimal_separator",
    )
    max_phases = st.sidebar.number_input(
        TEXTS["max_phases"][lang], 1, 10, 5, key="max_phases_input"
    )
    method = st.sidebar.selectbox(
        TEXTS["outlier_method_label"][lang],
        ["none", "simple", "rout"],
        index=2,
        format_func=lambda k: TEXTS["outlier_" + k][lang],
        key="outlier_method_selector",
    )
    q = (
        st.sidebar.slider(TEXTS["rout_q_label"][lang], 0.1, 10.0, 1.0, 0.1, key="rout_q_slider")
        if method == "rout"
        else 1.0
    )
    force_yi = st.sidebar.checkbox(TEXTS["force_yi"][lang], key="force_yi_checkbox")
    st.sidebar.caption(TEXTS["scope"][lang])
    seed_mode = st.sidebar.selectbox(
        TEXTS["seed_mode_label"][lang],
        ["fixed_42", "random"],
        format_func=lambda k: (
            TEXTS["seed_" + k][lang] if k == "fixed_42" else TEXTS["seed_random"][lang]
        ),
        key="seed_mode_selector",
    )
    st.sidebar.subheader(TEXTS["graphic_options_header"][lang])
    st.sidebar.selectbox(
        TEXTS["plot_font_label"][lang], PLOT_FONT_OPTIONS, key="plot_font_selector"
    )
    x_raw = st.sidebar.text_input(
        TEXTS["axis_x_custom_label"][lang], max_chars=200, key="axis_x_custom"
    )
    y_raw = st.sidebar.text_input(
        TEXTS["axis_y_custom_label"][lang], max_chars=200, key="axis_y_custom"
    )
    st.sidebar.caption(TEXTS["axis_math_hint"][lang])
    x_label = _axis(x_raw, TEXTS["axis_time"][lang], lang)
    y_label = _axis(y_raw, TEXTS["default_y_label"][lang], lang)
    if st.session_state.get("example_active"):
        content, filename = _example(), "synthetic_example.csv"
    elif uploaded is not None:
        content, filename = uploaded.getvalue(), uploaded.name
    else:
        content, filename = b"", ""
    if not content:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_key", None)
        st.info(TEXTS["info_upload"][lang])
    else:
        _data_workflow(
            content,
            filename,
            delimiter,
            decimal,
            max_phases,
            method,
            q,
            force_yi,
            seed_mode,
            lang,
            x_label,
            y_label,
        )
    with st.expander(TEXTS["instructions_header"][lang]):
        st.markdown(TEXTS["instructions_list"][lang])
    render_attribution(lang)
    st.caption(BUILD)


def _data_workflow(
    content,
    filename,
    delimiter,
    decimal,
    max_phases,
    method,
    q,
    force_yi,
    seed_mode,
    lang,
    x_label,
    y_label,
):
    seed = 42 if seed_mode == "fixed_42" else st.session_state.get("run_seed", 42)
    config = AnalysisConfig(max_phases, method, q, force_yi, seed)
    key = (
        analysis_key(content, config, delimiter, decimal)
        + filename.lower().rsplit(".", 1)[-1]
        + seed_mode
    )
    previous = st.session_state.get("analysis_key")
    if previous != key:
        st.session_state.pop("analysis_result", None)
        if previous:
            st.info(TEXTS["stale"][lang])
    st.session_state["analysis_key"] = key
    try:
        t, y, replicates, detected = read_upload(
            content, filename, delimiter=delimiter, decimal=decimal, return_format=True
        )
    except InputDataError as exc:
        st.session_state.pop("analysis_result", None)
        st.error(
            TEXTS.get(exc.code, TEXTS["file_parse"])[lang]
            + (" (" + exc.location + ")" if exc.location else "")
        )
        return
    with PLOT_LOCK:
        plot_raw_data(replicates, lang, x_label, y_label)
    st.success(TEXTS["data_loaded"][lang].format(len(replicates), len(t)))
    if detected["delimiter"] != "xlsx":
        st.caption(
            TEXTS["detected_format"][lang].format(
                "Tab" if detected["delimiter"] == "\t" else detected["delimiter"],
                detected["decimal"],
            )
        )
    if st.button(TEXTS["run_button"][lang], key="run_analysis_button"):
        st.session_state.pop("analysis_result", None)
        if seed_mode == "random":
            seed = secrets.randbelow(2_147_483_647)
            config = AnalysisConfig(max_phases, method, q, force_yi, seed)
        st.session_state["run_seed"] = seed
        st.session_state["analysis_key"] = (
            analysis_key(content, config, delimiter, decimal)
            + filename.lower().rsplit(".", 1)[-1]
            + seed_mode
        )
        try:
            started = time.monotonic()
            progress = st.progress(0.0, text=TEXTS["optimizing"][lang].format(max_phases))
            preview = st.empty()
            completed_rows = []

            def show_progress(name, phases, completed, candidate):
                progress.progress(
                    completed / (len(MODELS) * max_phases),
                    text=TEXTS["progress"][lang].format(
                        name, phases, max_phases, time.monotonic() - started
                    ),
                )
                if candidate is not None:
                    completed_rows.append(
                        {
                            TEXTS["full_model_col"][lang]: get_full_model_label(
                                candidate, name, lang
                            ),
                            "F": phases,
                            "y_i": format_estimate(
                                candidate["theta"][0], candidate["se"][0], lang, fixed=force_yi
                            ),
                            "y_f": format_estimate(candidate["theta"][1], candidate["se"][1], lang),
                            "SSE": candidate["metrics"]["SSE"],
                        }
                    )
                    preview.dataframe(pd.DataFrame(completed_rows), hide_index=True)

            try:
                result = run_analysis(t, y, config, on_progress=show_progress)
            finally:
                progress.empty()
                preview.empty()
            result["duration_seconds"] = time.monotonic() - started
            result["created_utc"] = datetime.now(UTC).isoformat()
            result["input_settings"] = {
                "delimiter": delimiter,
                "decimal": decimal,
                "resolved_delimiter": detected["delimiter"],
                "resolved_decimal": detected["decimal"],
            }
            st.session_state["analysis_result"] = result
        except AnalysisBusy:
            st.warning(TEXTS["busy"][lang])
        except Exception as exc:  # noqa: BLE001 - isolate failures without exposing input data
            LOGGER.error("Legacy analysis failed (%s)", type(exc).__name__)
            st.error(TEXTS["unexpected"][lang])
    bundle = st.session_state.get("analysis_result")
    if bundle is not None:
        st.caption(TEXTS["seed_used_msg"][lang].format(bundle["config"]["seed"]))
        if "duration_seconds" in bundle:
            st.caption(TEXTS["analysis_duration"][lang].format(bundle["duration_seconds"]))
        _render_results(
            bundle, replicates, lang, x_label, y_label, hashlib.sha256(content).hexdigest()
        )
