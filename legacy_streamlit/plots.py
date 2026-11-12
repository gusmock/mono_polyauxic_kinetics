"""Session-local figures for the public application."""

import io
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

from .core import (
    boltzmann_term_eq31,
    evaluate_polyauxic_model,
    first_order_term_phase1,
    gompertz_term_eq32,
)
from .i18n import TEXTS
from .labels import get_full_model_label
from .plots_data import observation_frame as build_all_data_df

PLOT_LOCK = threading.RLock()
AVAILABLE_MPL_FONT_NAMES = {font.name for font in fm.fontManager.ttflist}


def _first_available_font(candidates):
    for candidate in candidates:
        if candidate in AVAILABLE_MPL_FONT_NAMES:
            return candidate
    return None


DEFAULT_PLOT_FONT = (
    _first_available_font(["DejaVu Serif", "DejaVu Sans", "Arial", "Calibri", "Verdana"])
    or "DejaVu Sans"
)
DEFAULT_SANS_FONT = (
    _first_available_font(["DejaVu Sans", "Arial", "Calibri", "Verdana"]) or DEFAULT_PLOT_FONT
)
DEFAULT_SERIF_FONT = (
    _first_available_font(["DejaVu Serif", "Book Antiqua", "Bookman Old Style", DEFAULT_PLOT_FONT])
    or DEFAULT_PLOT_FONT
)

_BASE_PLOT_FONT_OPTIONS = [
    "DejaVu Serif",
    "DejaVu Sans",
    "Arial",
    "Calibri",
    "Book Antiqua",
    "Bookman Old Style",
    "Verdana",
    "Times New Roman",
]
# Keep all requested fonts visible in the UI. Runtime fallback is handled in resolve_plot_font.
PLOT_FONT_OPTIONS = list(_BASE_PLOT_FONT_OPTIONS)
if DEFAULT_PLOT_FONT not in PLOT_FONT_OPTIONS:
    PLOT_FONT_OPTIONS.insert(0, DEFAULT_PLOT_FONT)
if DEFAULT_PLOT_FONT in PLOT_FONT_OPTIONS:
    PLOT_FONT_OPTIONS = [DEFAULT_PLOT_FONT] + [
        font for font in PLOT_FONT_OPTIONS if font != DEFAULT_PLOT_FONT
    ]


def resolve_plot_font(font_name):
    candidate = str(font_name or "").strip()
    if candidate in AVAILABLE_MPL_FONT_NAMES:
        return candidate
    semantic_fallbacks = {
        "Times New Roman": ["DejaVu Serif", "Book Antiqua", "Bookman Old Style", "DejaVu Sans"],
        "Book Antiqua": ["DejaVu Serif", "Times New Roman", "Bookman Old Style", "DejaVu Sans"],
        "Bookman Old Style": ["DejaVu Serif", "Book Antiqua", "Times New Roman", "DejaVu Sans"],
        "Arial": ["DejaVu Sans", "Calibri", "Verdana", "DejaVu Serif"],
        "Calibri": ["DejaVu Sans", "Arial", "Verdana", "DejaVu Serif"],
        "Verdana": ["DejaVu Sans", "Arial", "Calibri", "DejaVu Serif"],
        "DejaVu Serif": ["DejaVu Sans"],
        "DejaVu Sans": ["DejaVu Serif"],
    }
    for fallback in semantic_fallbacks.get(candidate, []):
        if fallback in AVAILABLE_MPL_FONT_NAMES:
            return fallback
    return DEFAULT_PLOT_FONT


def get_current_plot_font():
    selected = st.session_state.get("plot_font_selector", DEFAULT_PLOT_FONT)
    return resolve_plot_font(selected)


def style_axes_fonts(ax, font_name=None):
    font_name = resolve_plot_font(font_name or get_current_plot_font())
    if ax.get_title():
        ax.set_title(ax.get_title(), fontname=font_name)
    ax.set_xlabel(ax.get_xlabel(), fontname=font_name)
    ax.set_ylabel(ax.get_ylabel(), fontname=font_name)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(font_name)
    leg = ax.get_legend()
    if leg:
        for txt in leg.get_texts():
            txt.set_fontname(font_name)


def plot_raw_data(replicates, lang, x_label, y_label):
    """Plots raw data before analysis."""
    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    for rep in replicates:
        ax.scatter(rep["t"], rep["y"], facecolors="white", edgecolors="black", alpha=0.8, s=20)
    ax.set_title(TEXTS["raw_title"][lang], fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", alpha=0.3)
    style_axes_fonts(ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_final_summary(replicates, best_results, lang, x_label, y_label):
    """Plots raw data + best fits for Gompertz and Boltzmann."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, rep in enumerate(replicates):
        label_txt = "Data" if i == 0 else ""
        ax.scatter(
            rep["t"],
            rep["y"],
            facecolors="white",
            edgecolors="black",
            alpha=0.6,
            s=20,
            label=label_txt,
        )

    colors = {"Gompertz": "tab:blue", "Boltzmann": "tab:orange"}

    all_t = [r["t"].max() for r in replicates]
    t_max = max(all_t) if all_t else 1.0
    t_smooth = np.linspace(0, t_max, 300)

    for model_name, res in best_results.items():
        if res is None:
            continue
        func = gompertz_term_eq32 if model_name == "Gompertz" else boltzmann_term_eq31
        y_smooth = evaluate_polyauxic_model(
            t_smooth,
            res["theta"],
            func,
            res["n_phases"],
            use_first_order_phase1=bool(res.get("use_first_order_phase1", False)),
        )

        label = f"{model_name}: {res['n_phases']}"
        ax.plot(t_smooth, y_smooth, linewidth=2, color=colors.get(model_name, "black"), label=label)

    ax.set_title(TEXTS["fit_summary"][lang], fontsize=12)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    style_axes_fonts(ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_metrics_summary(results_list, model_name, lang):
    """Generates a summary chart of metrics vs phases."""
    phases = [r["n_phases"] for r in results_list]
    aic = [r["metrics"]["AIC"] for r in results_list]
    aicc = [r["metrics"]["AICc"] for r in results_list]
    bic = [r["metrics"]["BIC"] for r in results_list]
    r2_adj = [r["metrics"]["R2_adj"] for r in results_list]
    phase1_structure = [bool(r.get("use_first_order_phase1", False)) for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(phases, aic, "o--", label="AIC")
    ax1.plot(phases, aicc, "s-", label="AICc")
    ax1.plot(phases, bic, "^:", label="BIC")
    for x, y, is_first_order in zip(phases, aicc, phase1_structure, strict=False):
        marker = "D" if is_first_order else "o"
        face = "tab:blue" if is_first_order else "white"
        ax1.scatter([x], [y], marker=marker, s=70, c=face, edgecolors="black", zorder=5)
    ax1.set_xlabel("Number of Phases")
    ax1.set_ylabel("Value")
    ax1.set_title("Information Criteria")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(phases, r2_adj, "o-", label="Adjusted R²")
    for x, y, is_first_order in zip(phases, r2_adj, phase1_structure, strict=False):
        marker = "D" if is_first_order else "o"
        face = "tab:blue" if is_first_order else "white"
        ax2.scatter([x], [y], marker=marker, s=70, c=face, edgecolors="black", zorder=5)
    ax2.set_xlabel("Number of Phases")
    ax2.set_ylabel("Adjusted R²")
    ax2.set_title("Fit Quality")
    marker_legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label=TEXTS["metrics_marker_sigmoidal"][lang],
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="None",
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            label=TEXTS["metrics_marker_first_order"][lang],
        ),
    ]
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + marker_legend)
    ax2.grid(True, alpha=0.3)
    style_axes_fonts(ax1)
    style_axes_fonts(ax2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    st.download_button(
        label=TEXTS["download_summary"][lang],
        data=buf.getvalue(),
        file_name=f"metrics_summary_{model_name}.svg",
        mime="image/svg+xml",
        on_click="ignore",
        key=f"dl_summary_{model_name}",
    )
    st.pyplot(fig)
    plt.close(fig)


def display_single_fit(
    res,
    replicates,
    model_name,
    model_func,
    color_main,
    x_label,
    y_label,
    param_labels,
    rate_label,
    lang,
):
    """Displays detailed results for a single fit."""
    n = res["n_phases"]
    theta = res["theta"]
    se = res["se"]
    se_p = res["se_p"]
    yi_name, yf_name = param_labels
    full_model_label = get_full_model_label(res, model_name, lang)
    phase1_label = (
        TEXTS["phase1_first_order"][lang]
        if res.get("use_first_order_phase1", False)
        else TEXTS["phase1_sigmoidal"][lang]
    )
    raw_data_w_outliers = build_all_data_df(replicates)
    outliers_mask = np.asarray(
        res.get("outliers", np.zeros(len(raw_data_w_outliers), dtype=bool)), dtype=bool
    )
    if len(outliers_mask) != len(raw_data_w_outliers):
        outliers_mask = np.zeros(len(raw_data_w_outliers), dtype=bool)
    raw_data_w_outliers["is_outlier"] = outliers_mask
    raw_data_w_outliers["t_round"] = raw_data_w_outliers["t"]
    stats_df = (
        raw_data_w_outliers[~raw_data_w_outliers["is_outlier"]]
        .groupby("t_round")["y"]
        .agg(["mean", "std"])
        .reset_index()
    )
    y_i, y_f = theta[0], theta[1]
    y_i_se, y_f_se = se[0], se[1]

    z = theta[2 : 2 + n]
    r_max = theta[2 + n : 2 + 2 * n]
    r_max_se = se[2 + n : 2 + 2 * n]
    lambda_ = theta[2 + 2 * n : 2 + 3 * n]
    lambda_se = se[2 + 2 * n : 2 + 3 * n]
    p = np.exp(z - np.max(z))
    p /= np.sum(p)

    use_first_order_phase1 = bool(res.get("use_first_order_phase1", False))
    phases = []
    for i in range(n):
        phases.append(
            {
                "phase_id": i,
                "p": p[i],
                "SE p": se_p[i],
                "r_max": r_max[i],
                "r_max_se": r_max_se[i],
                "lambda": lambda_[i],
                "lambda_se": lambda_se[i],
            }
        )
    # Keep phase-1 first when it is first-order (lambda_1 is not part of that equation).
    phases.sort(
        key=lambda x: (
            0 if (use_first_order_phase1 and x.get("phase_id", -1) == 0) else 1,
            x["lambda"],
        )
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for rep in replicates:
        ax.scatter(
            rep["t"], rep["y"], facecolors="white", edgecolors="black", alpha=0.8, s=15, marker="o"
        )

    outliers = raw_data_w_outliers[raw_data_w_outliers["is_outlier"]]
    if not outliers.empty:
        ax.scatter(
            outliers["t"],
            outliers["y"],
            color="red",
            marker="x",
            s=50,
            label=TEXTS["legend_outlier"][lang],
            zorder=5,
        )

    if len(replicates) > 1 and not stats_df.empty:
        ax.errorbar(
            stats_df["t_round"],
            stats_df["mean"],
            yerr=stats_df["std"],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=3,
            label=TEXTS["legend_mean"][lang],
            zorder=4,
        )

    t_max_val = raw_data_w_outliers["t"].max()
    t_smooth = np.linspace(0, t_max_val, 300)
    y_smooth = evaluate_polyauxic_model(
        t_smooth,
        theta,
        model_func,
        n,
        use_first_order_phase1=bool(res.get("use_first_order_phase1", False)),
    )
    ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label=TEXTS["legend_global"][lang])

    colors = plt.cm.viridis(np.linspace(0, 0.9, n))
    for i, ph in enumerate(phases):
        if use_first_order_phase1 and ph.get("phase_id", -1) == 0:
            y_ind = first_order_term_phase1(t_smooth, ph["p"], y_f - y_i, ph["r_max"])
        else:
            y_ind = model_func(t_smooth, y_i, y_f, ph["p"], ph["r_max"], ph["lambda"])
        y_vis = y_i + (y_f - y_i) * y_ind
        ax.plot(
            t_smooth,
            y_vis,
            "--",
            color=colors[i],
            alpha=0.6,
            label=TEXTS["legend_phase"][lang].format(i + 1),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.3)

    buf = io.BytesIO()
    style_axes_fonts(ax)
    fig.savefig(buf, format="svg")
    download_key = f"dl_btn_{model_name.lower()}_{model_func.__name__}_{n}_{'firstorder' if res.get('use_first_order_phase1', False) else 'standard'}"

    left_col, right_col = st.columns([2.4, 1.2], gap="large")
    with left_col:
        st.download_button(
            label=TEXTS["download_plot"][lang],
            data=buf.getvalue(),
            file_name=f"plot_{n}_phases.svg",
            mime="image/svg+xml",
            on_click="ignore",
            key=download_key,
        )
        st.pyplot(fig)
        plt.close(fig)

    m = res["metrics"]

    def _fmt_mixed(v):
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            return f"{float(v):.4f}"
        if isinstance(v, (int, float, np.integer, np.floating)) and not np.isfinite(v):
            return "N/A"
        return v

    with right_col:
        st.markdown(f"**{TEXTS['fit_overview_header'][lang]}**")
        df_overview = pd.DataFrame(
            {
                TEXTS["table_col_metric"][lang]: [
                    TEXTS["full_model_col"][lang],
                    TEXTS["phase1_col"][lang],
                    "F",
                    TEXTS["metric_corr"][lang],
                    "R²",
                    "R² Adj",
                    "SSE",
                    "AIC",
                    "AICc",
                    "BIC",
                ],
                TEXTS["table_col_value"][lang]: [
                    full_model_label,
                    phase1_label,
                    n,
                    m.get("r", np.nan),
                    m["R2"],
                    m["R2_adj"],
                    m["SSE"],
                    m["AIC"],
                    m["AICc"],
                    m["BIC"],
                ],
            }
        )
        df_overview[TEXTS["table_col_value"][lang]] = df_overview[
            TEXTS["table_col_value"][lang]
        ].map(_fmt_mixed)
        st.dataframe(df_overview, hide_index=True, width="stretch")

    with st.expander(TEXTS["details_expander"][lang], expanded=False):

        def _fmt_pm(v, se_v):
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                if isinstance(se_v, (int, float, np.integer, np.floating)) and np.isfinite(se_v):
                    return f"{float(v):.4f} ± {float(se_v):.4f}"
                return f"{float(v):.4f}"
            return str(v)

        df_glob = pd.DataFrame(
            {
                TEXTS["table_col_param"][lang]: [yi_name, yf_name],
                f"{TEXTS['table_col_value'][lang]} ± {TEXTS['table_col_se'][lang]}": [
                    _fmt_pm(y_i, y_i_se),
                    _fmt_pm(y_f, y_f_se),
                ],
            }
        )
        st.dataframe(df_glob, hide_index=True, width="stretch")

        rows = []
        for i, ph in enumerate(phases):
            lambda_display = (
                "N/A"
                if (use_first_order_phase1 and ph.get("phase_id", -1) == 0)
                else _fmt_pm(ph["lambda"], ph["lambda_se"])
            )
            rows.append(
                {
                    TEXTS["table_col_phase"][lang]: i + 1,
                    "p ± SE": _fmt_pm(ph["p"], ph["SE p"]),
                    f"{rate_label} ± {TEXTS['table_col_se'][lang]}": _fmt_pm(
                        ph["r_max"], ph["r_max_se"]
                    ),
                    "λ ± SE": lambda_display,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
