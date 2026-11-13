"""Bounded numerical orchestration and same-observation model selection."""

import json
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from . import BUILD
from .core import (
    AnalysisTimeout,
    boltzmann_term_eq31,
    choose_information_criterion,
    detect_outliers,
    detect_outliers_rout_rigorous,
    evaluate_polyauxic_model,
    fit_model_auto,
    fit_model_auto_robust_pre,
    gompertz_term_eq32,
    parameter_count,
    select_first_local_min_index,
)

ANALYSIS_SLOTS = threading.BoundedSemaphore(2)


class AnalysisBusy(RuntimeError):
    """The anonymous service is already running two analyses."""


MODELS = {"Gompertz": gompertz_term_eq32, "Boltzmann": boltzmann_term_eq31}


@dataclass(frozen=True)
class AnalysisConfig:
    max_phases: int = 5
    outlier_method: str = "rout"
    rout_q: float = 1.0
    force_yi: bool = False
    seed: int = 42

    def __post_init__(self):
        if type(self.max_phases) is not int or not 1 <= self.max_phases <= 10:
            raise ValueError("Invalid phase limit")
        if self.outlier_method not in ("none", "simple", "rout"):
            raise ValueError("Invalid outlier method")
        if not 0 < self.rout_q < 100:
            raise ValueError("Invalid ROUT Q")


def analysis_key(content, config, delimiter="auto", decimal="auto"):
    settings = json.dumps([BUILD, asdict(config), delimiter, decimal], sort_keys=True).encode()
    return sha256(content + b"\0" + settings).hexdigest()


def should_test_first_order_variant(result, tol=1e-6):
    n = result["n_phases"]
    return abs(float(result["theta"][2 + 2 * n])) <= tol


def run_fit_with_outlier_strategy(
    t_flat,
    y_flat,
    model_func,
    n,
    outlier_method_key,
    force_yi=False,
    force_yf=False,
    rout_q=1.0,
    use_first_order_phase1=False,
    random_seed=42,
    deadline=None,
):
    if outlier_method_key not in ("none", "simple", "rout"):
        raise ValueError("Unknown outlier method")
    options = {
        "force_yi": force_yi,
        "force_yf": force_yf,
        "use_first_order_phase1": use_first_order_phase1,
        "random_seed": random_seed,
        "deadline": deadline,
    }
    mask = np.zeros(len(y_flat), dtype=bool)
    pre = None
    if outlier_method_key != "none":
        pre_fit = fit_model_auto_robust_pre if outlier_method_key == "rout" else fit_model_auto
        pre = pre_fit(t_flat, y_flat, model_func, n, **options)
        if pre is None or not pre["converged"]:
            return None
        k = parameter_count(n, force_yi, force_yf, use_first_order_phase1)
        if outlier_method_key == "rout":
            proposed = detect_outliers_rout_rigorous(y_flat, pre["y_pred"], n_params=k, Q=rout_q)
        else:
            proposed = detect_outliers(y_flat, pre["y_pred"])
        if proposed.any() and np.count_nonzero(~proposed) > k + 5:
            mask = proposed
    if pre is not None and outlier_method_key == "simple" and not mask.any():
        result = pre
    else:
        result = fit_model_auto(t_flat[~mask], y_flat[~mask], model_func, n, **options)
    if result is None:
        return None
    result.update(
        outliers=mask,
        t_clean=t_flat[~mask],
        y_clean=y_flat[~mask],
        y_pred_full=evaluate_polyauxic_model(
            t_flat, result["theta"], model_func, n, use_first_order_phase1
        ),
    )
    return result


def get_best_candidate_by_phase(results, ic_name):
    grouped = {}
    for result in results:
        if np.isfinite(result["metrics"][ic_name]):
            grouped.setdefault(result["n_phases"], []).append(result)
    return [min(grouped[n], key=lambda r: r["metrics"][ic_name]) for n in sorted(grouped)]


def select_candidates(results):
    if not results:
        return None, [], None, "no_fit"
    eligible = [
        r for r in results if r.get("converged") and r.get("jacobian_rank") == r["n_parameters"]
    ]
    if not eligible:
        return None, [], None, "no_convergence"
    mask = eligible[0]["outliers"]
    # The same N alone is insufficient: identities of the retained observations must match.
    if any(not np.array_equal(mask, r["outliers"]) for r in eligible[1:]):
        return None, [], None, "different_observations"
    criterion = choose_information_criterion(
        eligible[0]["metrics"]["N"], max(r["metrics"]["k_IC"] for r in eligible)
    )
    by_phase = get_best_candidate_by_phase(eligible, criterion)
    if not by_phase:
        return None, [], criterion, "insufficient_ic"
    index = select_first_local_min_index([r["metrics"][criterion] for r in by_phase])
    return by_phase[index], by_phase, criterion, None


def _run_analysis(t, y, config, *, time_budget=None, on_progress=None):
    # Preserve completion of the requested candidates on slower institutional CPUs.
    # Each optimizer retains its 120 s limit; a caller may set an overall budget.
    deadline = time.monotonic() + time_budget if time_budget is not None else None
    candidates = {name: [] for name in MODELS}
    warnings = []
    first_order = None
    timed_out = False
    for family_index, (name, function) in enumerate(MODELS.items()):
        for n in range(1, config.max_phases + 1):
            completed = family_index * config.max_phases + n - 1
            if on_progress is not None:
                on_progress(name, n, completed, None)
            options = {
                "force_yi": config.force_yi,
                "rout_q": config.rout_q,
                "random_seed": config.seed,
                "deadline": deadline,
            }
            try:
                standard = run_fit_with_outlier_strategy(
                    t, y, function, n, config.outlier_method, **options
                )
                if standard is None:
                    warnings.append((name, n, "no_fit"))
                    continue
                candidates[name].append(standard)
                if on_progress is not None:
                    on_progress(name, n, completed, standard)
                if should_test_first_order_variant(standard, tol=1e-6):
                    if n == 1 and first_order is not None:
                        hybrid = deepcopy(first_order)
                    else:
                        hybrid = run_fit_with_outlier_strategy(
                            t,
                            y,
                            function,
                            n,
                            config.outlier_method,
                            use_first_order_phase1=True,
                            **options,
                        )
                    if hybrid is not None:
                        candidates[name].append(hybrid)
                        if on_progress is not None:
                            on_progress(name, n, completed, hybrid)
                        if n == 1:
                            first_order = deepcopy(hybrid)
            except AnalysisTimeout:
                warnings.append((name, n, "timeout"))
                timed_out = True
                break
            if on_progress is not None:
                on_progress(name, n, completed + 1, None)
        if timed_out:
            break
    families = {}
    for name, results in candidates.items():
        selected, phases, criterion, warning = select_candidates(results)
        if timed_out:
            selected, warning = None, "timeout"
        families[name] = {
            "results": results,
            "selected": selected,
            "by_phase": phases,
            "criterion": criterion,
            "warning": warning,
        }
    return {"families": families, "config": asdict(config), "warnings": warnings}


def run_analysis(t, y, config, *, time_budget=None, on_progress=None):
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        raise AnalysisBusy("The service is busy; retry later")
    try:
        return _run_analysis(t, y, config, time_budget=time_budget, on_progress=on_progress)
    finally:
        ANALYSIS_SLOTS.release()
