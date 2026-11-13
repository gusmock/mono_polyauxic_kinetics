"""Excel export; all generation is in memory."""

import io

import numpy as np
import pandas as pd

from .labels import boundary_parameter_labels, get_full_model_label


def generate_excel_report(
    best_results_global, replicates, param_labels, rate_label, lang, *, analysis=None, metadata=None
):
    """
    Generates an Excel file (.xlsx) in memory containing all parameters,
    standard errors, and statistics for the best models found.
    """
    output = io.BytesIO()

    # Use xlsxwriter engine to format Excel in memory
    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}},
    ) as writer:
        import json

        from .plots_data import observation_frame

        raw = observation_frame(replicates)
        raw.to_excel(writer, sheet_name="Data", index=False)
        audit = dict(metadata or {})
        if analysis is not None:
            audit.update(analysis["config"])
            audit.update(analysis.get("input_settings", {}))
            audit["created_utc"] = analysis.get("created_utc", "")
            audit["duration_seconds"] = analysis.get("duration_seconds", "")
            audit["warnings"] = json.dumps(analysis["warnings"])
            candidates = []
            for name, family in analysis["families"].items():
                audit[name + "_criterion"] = family["criterion"]
                audit[name + "_selection_warning"] = family["warning"]
                for res in family["results"]:
                    hybrid = res["use_first_order_phase1"]
                    identifier = f"{name[:3]}_{res['n_phases']}_{'FO' if hybrid else 'S'}"
                    candidates.append(
                        {
                            "candidate": identifier,
                            "family": name,
                            "phases": res["n_phases"],
                            "first_order": hybrid,
                            "selected": res is family["selected"],
                            "converged": res["converged"],
                            "jacobian_rank": res["jacobian_rank"],
                            "uncertainty_reliable": res["uncertainty_reliable"],
                            "uncertainty_status": res.get("uncertainty_status", ""),
                            "jacobian_condition": res.get("jacobian_condition", np.nan),
                            "active_constraints": "; ".join(boundary_parameter_labels(res)),
                            "search_strategy": res.get("search_strategy", ""),
                            "optimizer_refinement": res.get("optimizer_refinement", ""),
                            **res["metrics"],
                        }
                    )
                    predicted = raw.copy()
                    predicted["predicted"] = res["y_pred_full"]
                    predicted["residual"] = predicted["y"] - predicted["predicted"]
                    predicted["excluded"] = res["outliers"]
                    predicted.to_excel(writer, sheet_name=identifier + "_Data", index=False)
                    n = res["n_phases"]
                    theta, se = res["theta"], res["se"]
                    weights = np.exp(theta[2 : 2 + n] - np.max(theta[2 : 2 + n]))
                    weights /= weights.sum()
                    parameters = [
                        {"parameter": "y_i", "phase": None, "value": theta[0], "SE": se[0]},
                        {"parameter": "y_f", "phase": None, "value": theta[1], "SE": se[1]},
                    ]
                    for j in range(n):
                        parameters += [
                            {
                                "parameter": "p",
                                "phase": j + 1,
                                "value": weights[j],
                                "SE": res["se_p"][j],
                            },
                            {
                                "parameter": "r_max",
                                "phase": j + 1,
                                "value": theta[2 + n + j],
                                "SE": se[2 + n + j],
                            },
                            {
                                "parameter": "lambda",
                                "phase": j + 1,
                                "value": np.nan if hybrid and j == 0 else theta[2 + 2 * n + j],
                                "SE": se[2 + 2 * n + j],
                            },
                        ]
                    pd.DataFrame(parameters).to_excel(
                        writer, sheet_name=identifier + "_Parameters", index=False
                    )
            pd.DataFrame(candidates).to_excel(writer, sheet_name="Candidates", index=False)
        pd.DataFrame({"key": list(audit), "value": list(audit.values())}).to_excel(
            writer, sheet_name="Audit", index=False
        )

        # Loop through the best results dictionary (Gompertz, Boltzmann)
        for model_name, res in best_results_global.items():
            if res is None:
                continue

            n = res["n_phases"]
            theta = res["theta"]
            se = res["se"]
            se_p = res["se_p"]
            yi_name, yf_name = param_labels
            model_full_label = get_full_model_label(res, model_name, lang)
            use_first_order_phase1 = bool(res.get("use_first_order_phase1", False))

            def _fmt_pm_excel(v, se_v):
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                    if isinstance(se_v, (int, float, np.integer, np.floating)) and np.isfinite(
                        se_v
                    ):
                        return f"{float(v):.6g} ± {float(se_v):.6g}"
                    return f"{float(v):.6g} ± N/A"
                return str(v)

            # --- Global Parameters Sheet ---
            global_params = pd.DataFrame(
                {
                    "Parameter": [
                        yi_name,
                        yf_name,
                        "Phases (n)",
                        "Phase 1 Structure",
                        "Full Model",
                    ],
                    "Value ± SE": [
                        _fmt_pm_excel(theta[0], se[0]),
                        _fmt_pm_excel(theta[1], se[1]),
                        n,
                        "First-order" if res.get("use_first_order_phase1", False) else "Sigmoidal",
                        model_full_label,
                    ],
                }
            )
            global_params.to_excel(writer, sheet_name=f"{model_name}_Global", index=False)

            # --- Phase Specific Parameters Sheet ---
            z = theta[2 : 2 + n]
            r_max = theta[2 + n : 2 + 2 * n]
            r_max_se = se[2 + n : 2 + 2 * n]
            lambda_ = theta[2 + 2 * n : 2 + 3 * n]
            lambda_se = se[2 + 2 * n : 2 + 3 * n]

            # Calculate proportion 'p' based on z transformation
            p = np.exp(z - np.max(z))
            p /= np.sum(p)

            phase_data = []
            for i in range(n):
                is_first_order_phase1 = use_first_order_phase1 and i == 0
                lambda_display = (
                    "N/A" if is_first_order_phase1 else _fmt_pm_excel(lambda_[i], lambda_se[i])
                )
                lambda_order = (
                    float("-inf")
                    if is_first_order_phase1
                    else (float(lambda_[i]) if np.isfinite(lambda_[i]) else np.inf)
                )
                phase_data.append(
                    {
                        "Phase": i + 1,
                        "Full Model": model_full_label,
                        "p ± SE": _fmt_pm_excel(p[i], se_p[i]),
                        f"{rate_label} ± SE": _fmt_pm_excel(r_max[i], r_max_se[i]),
                        "lambda ± SE": lambda_display,
                        "_lambda_order": lambda_order,
                    }
                )

            phase_df = pd.DataFrame(phase_data)
            # Sort phases sequentially by lambda (lag time)
            phase_df = (
                phase_df.sort_values(by="_lambda_order")
                .drop(columns=["_lambda_order"])
                .reset_index(drop=True)
            )
            phase_df.to_excel(writer, sheet_name=f"{model_name}_Phases", index=False)

            # --- Statistics & Metrics Sheet ---
            m = res["metrics"]
            metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Full Model",
                        "Correlation (r)",
                        "R-squared",
                        "Adjusted R-squared",
                        "SSE",
                        "AIC",
                        "AICc",
                        "BIC",
                    ],
                    "Value": [
                        model_full_label,
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
            metrics_df.to_excel(writer, sheet_name=f"{model_name}_Metrics", index=False)

    return output.getvalue()
