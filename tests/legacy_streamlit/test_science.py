import numpy as np
import pytest

from legacy_streamlit import core
from legacy_streamlit.analysis import (
    AnalysisConfig,
    analysis_key,
    run_analysis,
    run_fit_with_outlier_strategy,
    select_candidates,
)


@pytest.mark.parametrize("function", [core.gompertz_term_eq32, core.boltzmann_term_eq31])
def test_one_phase_known_parameters(function):
    t = np.linspace(0, 15, 30)
    theta = np.array([1.0, 12.0, 0.0, 2.0, 2.0])
    y = core.polyauxic_model(t, theta, function, 1)
    result = core.fit_model_auto(t, y, function, 1)
    assert result["converged"]
    np.testing.assert_allclose(result["y_pred"], y, atol=3e-5)
    np.testing.assert_allclose(
        result["theta"][[0, 1, 3, 4]], theta[[0, 1, 3, 4]], rtol=1e-4, atol=1e-4
    )
    assert result["metrics"]["k_model"] == 4
    assert result["metrics"]["k_IC"] == 5
    assert result["jacobian_rank"] == 4


def test_first_order_amplitude_and_initial_rate():
    t = np.array([0.0, 1e-6, 1.0, 100.0])
    theta = np.array([5.0, 15.0, 0.0, 2.0, 0.0])
    y = core.evaluate_polyauxic_model(t, theta, core.gompertz_term_eq32, 1, True)
    np.testing.assert_allclose(y, 5 + 10 * (-np.expm1(-0.2 * t)))
    assert (y[1] - y[0]) / t[1] == pytest.approx(2.0, rel=1e-6)
    assert core.parameter_count(1, use_first_order_phase1=True) == 3


def test_hybrid_zero_initial_asymptote_is_exact_fixed_not_estimated():
    t = np.linspace(0, 20, 40)
    theta = np.array([0.0, 10.0, 0.0, 1.8, 0.0])
    y = core.evaluate_polyauxic_model(t, theta, core.gompertz_term_eq32, 1, True)
    r = core.fit_model_auto(
        t, y, core.gompertz_term_eq32, 1, force_yi=True, use_first_order_phase1=True
    )
    np.testing.assert_allclose(r["y_pred"], y, atol=1e-4)
    assert r["theta"][0] == 0 and r["theta"][4] == 0
    assert r["se"][0] == 0 and np.isnan(r["se"][4])
    assert r["metrics"]["k_model"] == 2 and r["metrics"]["k_IC"] == 3


def test_metric_formula_counts_variance_and_free_parameters():
    y = np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])
    pred = y + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    m = core.calculate_metrics(y, pred, 2)
    assert m["SSE"] == pytest.approx(0.06)
    assert m["AIC"] == pytest.approx(6 * np.log(0.06 / 6) + 2 * 3)
    assert m["AICc"] == pytest.approx(m["AIC"] + 2 * 3 * 4 / (6 - 3 - 1))
    assert m["R2_adj"] == pytest.approx(1 - (1 - m["R2"]) * 5 / 4)


def candidate(mask, score=1.0, n=1, first_order=False):
    return {
        "n_phases": n,
        "n_parameters": 4,
        "converged": True,
        "jacobian_rank": 4,
        "outliers": np.array(mask),
        "use_first_order_phase1": first_order,
        "metrics": {"N": 20, "k_IC": 5, "AICc": score, "AIC": score, "BIC": score},
    }


def test_equal_counts_with_different_excluded_rows_are_not_comparable():
    best, _, _, reason = select_candidates([candidate([True, False]), candidate([False, True])])
    assert best is None and reason == "different_observations"


def test_first_local_minimum_policy_preserved():
    rows = [candidate([False], score, n) for n, score in enumerate([10.0, 8.0, 9.0, 1.0], 1)]
    best, _, _, reason = select_candidates(rows)
    assert reason is None and best["n_phases"] == 2


def test_nonfinite_ic_and_nonconvergence_never_selected():
    r = candidate([False], np.inf)
    assert select_candidates([r])[0] is None
    r = candidate([False])
    r["converged"] = False
    assert select_candidates([r])[0] is None


def test_fingerprint_includes_scientific_and_parse_settings():
    c = AnalysisConfig()
    assert analysis_key(b"a", c) != analysis_key(b"b", c)
    assert analysis_key(b"a", c) != analysis_key(b"a", AnalysisConfig(max_phases=2))
    assert analysis_key(b"a", c) != analysis_key(b"a", c, decimal=",")


def test_new_build_invalidates_previous_numerical_results(monkeypatch):
    from legacy_streamlit import analysis

    previous = analysis_key(b"data", AnalysisConfig())
    monkeypatch.setattr(analysis, "BUILD", "synthetic-next-build")
    assert analysis_key(b"data", AnalysisConfig()) != previous


def test_timeout_does_not_select_partial_results():
    t = np.linspace(0, 15, 25)
    y = 1 + 10 / (1 + np.exp(-t + 5))
    result = run_analysis(t, y, AnalysisConfig(max_phases=1), time_budget=-1)
    assert all(f["selected"] is None for f in result["families"].values())
    assert result["warnings"][0][2] == "timeout"


@pytest.mark.parametrize(
    "bad_t,bad_y", [([0, 1], [1, np.inf]), ([0, 0], [1, 2]), ([-1, 0], [1, 2]), ([0, 1], [2, 2])]
)
def test_invalid_arrays_fail_before_optimization(bad_t, bad_y):
    with pytest.raises(ValueError):
        core.fit_model_auto(bad_t, bad_y, core.gompertz_term_eq32, 1)


def test_final_zero_is_not_a_valid_decreasing_model():
    with pytest.raises(ValueError, match="zero final"):
        core.fit_model_auto(np.arange(10), np.arange(10), core.gompertz_term_eq32, 1, force_yf=True)


@pytest.mark.parametrize("function", [core.gompertz_term_eq32, core.boltzmann_term_eq31])
def test_two_phase_recovery_and_time_unit_invariance(function):
    t = np.linspace(0, 30, 65)
    theta = np.array([1.0, 20.0, np.log(0.4 / 0.6), 0.0, 1.5, 1.2, 2.0, 14.0])
    y = core.polyauxic_model(t, theta, function, 2)
    first = core.fit_model_auto(t, y, function, 2)
    converted = core.fit_model_auto(t * 24, y, function, 2)
    assert first["converged"] and converted["converged"]
    assert first["n_parameters"] == 7
    np.testing.assert_allclose(first["y_pred"], y, atol=2e-4)
    np.testing.assert_allclose(converted["y_pred"], first["y_pred"], atol=2e-4)
    np.testing.assert_allclose(converted["theta"][4:6] * 24, first["theta"][4:6], rtol=1e-3)
    np.testing.assert_allclose(converted["theta"][6:8] / 24, first["theta"][6:8], rtol=1e-3)


@pytest.mark.parametrize("method", ["simple", "rout"])
def test_outlier_refit_preserves_full_observation_index(method):
    t = np.linspace(0, 15, 45)
    y = core.polyauxic_model(t, [1.0, 12.0, 0.0, 2.0, 2.0], core.gompertz_term_eq32, 1)
    y += np.random.default_rng(10).normal(0, 0.03, len(t))
    y[25] += 6.0
    result = run_fit_with_outlier_strategy(t, y, core.gompertz_term_eq32, 1, method)
    assert result is not None
    assert result["outliers"][25]
    assert len(result["y_pred_full"]) == len(y)
    np.testing.assert_array_equal(result["y_clean"], y[~result["outliers"]])
    assert result["metrics"]["N"] == len(result["y_clean"])


def test_concurrency_limit_does_not_share_or_start_excess_jobs():
    from legacy_streamlit.analysis import ANALYSIS_SLOTS, AnalysisBusy

    assert ANALYSIS_SLOTS.acquire(blocking=False)
    assert ANALYSIS_SLOTS.acquire(blocking=False)
    try:
        with pytest.raises(AnalysisBusy):
            run_analysis(np.arange(10.0), np.arange(10.0), AnalysisConfig(max_phases=1))
    finally:
        ANALYSIS_SLOTS.release()
        ANALYSIS_SLOTS.release()


def test_progress_exposes_completed_candidates_before_final_selection():
    events = []
    t = np.linspace(0, 15, 25)
    y = core.polyauxic_model(t, [1.0, 12.0, 0.0, 2.0, 2.0], core.gompertz_term_eq32, 1)
    result = run_analysis(
        t, y, AnalysisConfig(max_phases=1), on_progress=lambda *event: events.append(event)
    )
    assert AnalysisConfig().outlier_method == "rout"
    assert events[0] == ("Gompertz", 1, 0, None)
    assert events[-1] == ("Boltzmann", 1, 2, None)
    completed = [(name, candidate) for name, _, _, candidate in events if candidate is not None]
    assert {name for name, _ in completed} == {"Gompertz", "Boltzmann"}
    for name, candidate in completed:
        assert any(candidate is r for r in result["families"][name]["results"])


def test_legacy_search_recovers_historical_noisy_three_phase_fit():
    # Public synthetic benchmark, not experimental data. Reference: original
    # public commit 2bc20a9, seed 42, same noisy two-phase signal fitted with 3 phases.
    t = np.linspace(0, 30, 65)
    theta = [1.0, 20.0, np.log(0.4 / 0.6), 0.0, 1.5, 1.2, 2.0, 14.0]
    y = core.polyauxic_model(t, theta, core.gompertz_term_eq32, 2)
    y += np.random.default_rng(10).normal(0, 0.08, len(t))
    result = run_fit_with_outlier_strategy(t, y, core.gompertz_term_eq32, 3, "rout")
    assert result is not None
    assert result["metrics"]["SSE"] == pytest.approx(0.21977902897374363, abs=5e-4)
    assert not result["outliers"].any()


def test_active_bound_retains_explicitly_approximate_errors_with_analytic_check():
    t = np.linspace(0, 4, 30)
    y = core.polyauxic_model(t, [5.0, 100.0, 0.0, 8.0, 2.0], core.boltzmann_term_eq31, 1)
    r = core.fit_model_auto(t, y, core.boltzmann_term_eq31, 1)
    assert r["converged"] and r["jacobian_rank"] == 4
    assert 1 in r["active_bounds"]  # yf reaches the legacy search bound.
    assert r["uncertainty_status"] == "approximate_at_boundary"
    assert not r["uncertainty_reliable"]
    assert np.isfinite(r["se"][[0, 1, 3, 4]]).all()
    # Independent analytic derivatives of the one-phase Boltzmann equation.
    yi, yf, _, rate, lag = r["theta"]
    delta = yf - yi
    a = 4 * rate * (lag - t) / delta
    s = 1 / (1 + np.exp(a + 2))
    q = s * (1 - s)
    jac = np.column_stack((1 - s - q * a, s + q * a, -4 * q * (lag - t), -4 * rate * q))
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    cov = r["metrics"]["SSE"] / (len(t) - 4) * (vt.T / singular**2) @ vt
    np.testing.assert_allclose(r["se"][[0, 1, 3, 4]], np.sqrt(np.diag(cov)), rtol=0.01)


def test_failed_legacy_refinement_recovers_with_explicit_lag_constraints(monkeypatch):
    from scipy.optimize import OptimizeResult

    t = np.linspace(0, 20, 45)
    truth = np.array([1.0, 12.0, 0.0, 0.0, 2.0, 1.0, 2.0, 7.0])
    y = core.polyauxic_model(t, truth, core.gompertz_term_eq32, 2)
    y += np.random.default_rng(7).normal(0, 0.02, len(t))
    initial = np.array([1.2, 11.5, 0.1, -0.1, 1.8, 1.2, 2.0, 5.0])
    initial /= np.r_[y.max(), y.max(), 1, 1, y.max() / t.max(), y.max() / t.max(), t.max(), t.max()]
    real_minimize = core.minimize

    def unsuccessful_search(fun, bounds, **kwargs):
        return OptimizeResult(
            x=initial.copy(), fun=fun(initial), success=False, message="iteration limit"
        )

    def failed_refinement(fun, x, **kwargs):
        if kwargs["method"] == "L-BFGS-B":
            return OptimizeResult(x=x.copy(), fun=fun(x), success=False, message="ABNORMAL")
        return real_minimize(fun, x, **kwargs)

    monkeypatch.setattr(core, "differential_evolution", unsuccessful_search)
    monkeypatch.setattr(core, "minimize", failed_refinement)
    result = core.fit_model_auto(t, y, core.gompertz_term_eq32, 2)
    assert result["converged"]
    assert result["optimizer_refinement"].startswith("SLSQP")
    assert np.all(np.diff(result["theta"][6:]) >= -1e-7)
    assert result["metrics"]["SSE"] < 0.05


@pytest.mark.parametrize("budget,complete", [(None, True), (180.0, False)])
def test_slow_complete_analysis_is_not_cut_off_without_explicit_overall_budget(
    monkeypatch, budget, complete
):
    from legacy_streamlit import analysis

    elapsed = [0.0]
    monkeypatch.setattr(analysis.time, "monotonic", lambda: elapsed[0])

    def slow_fit(t, y, function, n, method, **options):
        elapsed[0] += 100.0  # Each fit is within its own 120-second budget.
        deadline = options["deadline"]
        if deadline is not None and elapsed[0] > deadline:
            raise core.AnalysisTimeout("overall budget expired")
        r = candidate([False] * len(t), score=10 / n, n=n)
        r["theta"] = np.ones(2 + 3 * n)  # No optional first-order candidate.
        return r

    monkeypatch.setattr(analysis, "run_fit_with_outlier_strategy", slow_fit)
    r = run_analysis(
        np.arange(30.0), np.arange(30.0), AnalysisConfig(max_phases=2), time_budget=budget
    )
    if complete:
        assert elapsed[0] == 400.0
        assert all(
            len(f["results"]) == 2 and f["selected"] is not None for f in r["families"].values()
        )
    else:
        assert all(f["selected"] is None for f in r["families"].values())
        assert r["warnings"][-1][2] == "timeout"
