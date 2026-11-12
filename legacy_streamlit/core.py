"""Public scientific core. Reference equations: doi:10.1007/s11538-026-01621-7.

The returned theta layout remains [yi, yf, n logits, n rates, n lags].
Only independent parameters are optimized. The last logit is the zero reference;
fixed asymptotes and the unused first-order lag are excluded from estimation.
"""

import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from scipy.stats import t as t_dist

from .data import process_data  # noqa: F401 - public compatibility export


class AnalysisTimeout(RuntimeError):
    """The bounded computation budget expired; no partial fit is accepted."""


def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Boltzmann model term (Eq. 31).
    This represents a single sigmoidal phase of the modified Boltzmann equation.
    The parameters have direct physical and biological interpretations:
    y_i (initial asymptote), y_f (final asymptote), p_j (weighting factor for this phase),
    r_max_j (maximum specific reaction rate), and lambda_j (lag phase duration).
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe / (1.0 + np.exp(exponent))


def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Gompertz model term (Eq. 32).
    This represents a single phase of the modified Gompertz model. Unlike the symmetric
    Boltzmann equation, the Gompertz curve is inherently asymmetric, making it useful for
    processes with early rapid changes followed by slower sustained progression.
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe * np.exp(-np.exp(exponent))


def first_order_term_phase1(t, p_j, response_amplitude, r_max_j):
    """
    First-order phase contribution used when phase 1 starts immediately
    (|lambda_1| approximately zero).
    Returns a dimensionless term compatible with the global weighted sum.
    """
    t = np.asarray(t, dtype=float)
    p_safe = max(float(p_j), 1e-12)
    y_f_safe = abs(float(response_amplitude))
    if abs(y_f_safe) < 1e-12:
        y_f_safe = 1e-12
    k = r_max_j / (p_safe * y_f_safe)
    exponent = np.clip(-k * t, -500.0, 500.0)
    return p_safe * -np.expm1(exponent)


def polyauxic_model(t, theta, model_func, n_phases):
    """
    Global polyauxic model: Summation of weighted phases.
    To describe multiphasic (polyauxic) growth behaviors, the overall growth curve
    is modeled as a weighted sum of individual sigmoidal functions.
    """
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]

    # Softmax transformation for weights p (Eq. 33).
    # Ensures all weighting factors (p_j) are strictly positive and sum to 1,
    # preserving correct amplitude scaling.
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases


def polyauxic_model_phase1_first_order(t, theta, model_func, n_phases):
    """
    Hybrid polyauxic model where phase 1 is replaced by a first-order model:
    Phase contribution: p1*(1-exp(-r_max1*t/(p1*abs(y_f-y_i)))).
    """
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]

    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    if n_phases > 0:
        sum_phases += first_order_term_phase1(t, p[0], y_f - y_i, r_max[0])
    for j in range(1, n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases


def evaluate_polyauxic_model(t, theta, model_func, n_phases, use_first_order_phase1=False):
    """Dispatch helper for classic vs hybrid phase-1 model."""
    if use_first_order_phase1:
        return polyauxic_model_phase1_first_order(t, theta, model_func, n_phases)
    return polyauxic_model(t, theta, model_func, n_phases)


def calculate_p_errors(z_vals, cov_z):
    """
    Standard error propagation for p (Softmax weights).
    For derived parameters such as the phase fractions (p_j), standard errors are
    approximated using the Delta method by propagating the covariance of the latent
    variables (z) through the Jacobian of the Softmax transformation.
    """
    exps = np.exp(z_vals - np.max(z_vals))
    p = exps / np.sum(exps)
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, j] = p[i] * (1 - p[i])
            else:
                J[i, j] = -p[i] * p[j]
    try:
        cov_p = J @ cov_z @ J.T
        se_p = np.sqrt(np.maximum(0.0, np.diag(cov_p)))
        return se_p
    except (ValueError, np.linalg.LinAlgError):
        return np.full(n, np.nan)


def detect_outliers(y_true, y_pred):
    """Simple outlier detection based on MAD (Z-score > 2.5)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5


def detect_outliers_rout_rigorous(y_true, y_pred, n_params=None, Q=1.0):
    """
    ROUT (Rigorous) Method with FDR control.
    Identifies experimental outliers utilizing the False Discovery Rate (FDR) based on
    deviations from the robust pre-fitted model.
    Updated to handle optional n_params for backward compatibility.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n = residuals.size

    # If the function is called externally without n_params, assume 1 to avoid breaking the execution
    safe_n_params = n_params if n_params is not None else 1

    if n <= safe_n_params:
        return np.zeros_like(residuals, dtype=bool)

    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    rsdr = 1.4826 * mad_res if mad_res > 1e-12 else 1e-12

    t_scores = residuals / rsdr

    # Use n_params if provided (new method), otherwise default to the old method (n - 1)
    if n_params is not None:
        df = max(n - n_params, 1)
    else:
        df = max(n - 1, 1)

    abs_t = np.abs(t_scores)
    p_values = 2.0 * t_dist.sf(abs_t, df=df)

    if not 0 < Q < 100:
        raise ValueError("Q must be between 0 and 100 percent")
    alpha = Q / 100.0
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    i = np.arange(1, n + 1)
    bh_thresholds = (i / n) * alpha
    below = p_sorted <= bh_thresholds

    if not np.any(below):
        return np.zeros_like(residuals, dtype=bool)

    k_max = np.max(np.where(below)[0])
    p_crit = p_sorted[k_max]
    mask_outliers = p_values <= p_crit
    return mask_outliers


def smart_initial_guess(t, y, n_phases):
    """
    Heuristic initial parameter estimation using derivatives.
    Automatically analyzes the first derivative of the response data to identify peaks
    representing inflection points, ensuring global optimization begins within biologically
    plausible regions.
    """
    t_arr = np.asarray(t, dtype=float).flatten()
    y_arr = np.asarray(y, dtype=float).flatten()
    valid_mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[valid_mask]
    y_arr = y_arr[valid_mask]

    if t_arr.size == 0 or y_arr.size == 0:
        return np.zeros(2 + 3 * n_phases)

    sort_idx = np.argsort(t_arr)
    t_sorted = t_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    unique_t, inverse = np.unique(t_sorted, return_inverse=True)
    if unique_t.size != t_sorted.size:
        y_sum = np.zeros(unique_t.size, dtype=float)
        y_count = np.zeros(unique_t.size, dtype=float)
        np.add.at(y_sum, inverse, y_sorted)
        np.add.at(y_count, inverse, 1.0)
        y_unique = y_sum / np.maximum(y_count, 1.0)
    else:
        y_unique = y_sorted

    if unique_t.size > 1:
        dy = np.gradient(y_unique, unique_t, edge_order=1)
    else:
        dy = np.zeros_like(y_unique)

    smooth_window = min(5, max(1, unique_t.size))
    dy_smooth = np.convolve(dy, np.ones(smooth_window) / smooth_window, mode="same")
    min_dist = max(1, len(unique_t) // max(1, (n_phases * 4)))
    peak_height = np.max(dy_smooth) * 0.1 if dy_smooth.size else 0.0
    peaks, props = find_peaks(dy_smooth, height=peak_height, distance=min_dist)
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({"lambda": unique_t[p_idx], "r_max": dy_smooth[p_idx]})
    while len(guesses) < n_phases:
        t_span = max(unique_t.max() - unique_t.min(), 1e-9)
        guesses.append(
            {
                "lambda": unique_t.min() + t_span * (len(guesses) + 1) / (n_phases + 1),
                "r_max": (np.max(y_unique) - np.min(y_unique)) / (t_span / max(n_phases, 1)),
            }
        )
    guesses.sort(key=lambda x: x["lambda"])
    theta_guess = np.zeros(2 + 3 * n_phases)

    n_slice = max(1, len(y_unique) // 5)
    mean_start = np.mean(y_unique[:n_slice])
    mean_end = np.mean(y_unique[-n_slice:])

    if float(mean_start) < float(mean_end):
        theta_guess[0] = np.min(y_unique)
        theta_guess[1] = np.max(y_unique)
    else:
        theta_guess[0] = np.max(y_unique)
        theta_guess[1] = np.min(y_unique)

    theta_guess[2 : 2 + n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2 + n_phases + i] = guesses[i]["r_max"]
        theta_guess[2 + 2 * n_phases + i] = guesses[i]["lambda"]
    return theta_guess


def choose_information_criterion(N, k_max):
    """
    Selects AIC, AICc or BIC based on sample size N and parameters k.
    Proper model parsimony requires specific criteria to penalize overparameterization
    as the number of stacked phases (n) grows.
    """
    dof_ratio = N / max(k_max, 1)
    if N <= 200:
        if dof_ratio < 40:
            return "AICc"
        else:
            return "AIC"
    else:
        return "BIC"


def select_first_local_min_index(values, tol=1e-9):
    """
    Selects the first local minimum index.
    In the context of information criteria, the optimal number of phases is the smallest
    value for which the criterion reaches its minimum.
    """
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Selection requires nonempty, finite comparable criteria")
    best_idx = 0
    for i in range(1, len(values)):
        if values[i] < values[best_idx] - tol:
            best_idx = i
        elif values[i] >= values[best_idx] - tol:
            break
    return best_idx


def parameter_count(n_phases, force_yi=False, force_yf=False, use_first_order_phase1=False):
    """Independent mean-model parameters; Gaussian variance adds one IC parameter."""
    return 3 * int(n_phases) + 1 - int(force_yi) - int(force_yf) - int(use_first_order_phase1)


def _validate_fit_data(t, y, n_phases):
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or not len(t):
        raise ValueError("Time and response must be nonempty paired one-dimensional arrays")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Time and response must contain only finite values")
    if np.any(t < 0) or np.ptp(t) <= 0:
        raise ValueError("Nonnegative times and at least two distinct times are required")
    if not isinstance(n_phases, (int, np.integer)) or not 1 <= n_phases <= 10:
        raise ValueError("The number of phases must be an integer between 1 and 10")
    if np.ptp(y) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
        raise ValueError("Constant responses do not identify kinetic parameters")
    return t, y


def calculate_metrics(y, y_pred, k_model):
    """Gaussian likelihood, estimated residual variance included in IC parameter count.

    Additive likelihood constants common to candidates are omitted. Selection is
    only valid for exactly the same observations (enforced by analysis.py).
    """
    n = len(y)
    sse = float(np.sum((y - y_pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    r = float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y) > 0 and np.std(y_pred) > 0 else np.nan
    k_ic = k_model + 1
    log_likelihood = n * np.log(max(sse / n, np.finfo(float).tiny))
    aic = log_likelihood + 2 * k_ic
    return {
        "r": r,
        "R2": r2,
        "R2_adj": 1 - (1 - r2) * (n - 1) / (n - k_model) if n > k_model else np.nan,
        "SSE": sse,
        "AIC": aic,
        "BIC": log_likelihood + k_ic * np.log(n),
        "AICc": aic + 2 * k_ic * (k_ic + 1) / (n - k_ic - 1) if n > k_ic + 1 else np.inf,
        "N": n,
        "k_model": k_model,
        "k_IC": k_ic,
    }


def _fit(
    t_data,
    y_data,
    model_func,
    n_phases,
    *,
    force_yi=False,
    force_yf=False,
    use_first_order_phase1=False,
    random_seed=42,
    robust=False,
    deadline=None,
):
    t_data, y_data = _validate_fit_data(t_data, y_data, n_phases)
    if force_yf:
        raise ValueError(
            "A zero final asymptote is incompatible with this increasing-response model"
        )
    n = n_phases
    k = parameter_count(n, force_yi, force_yf, use_first_order_phase1)
    if len(t_data) <= k or len(np.unique(t_data)) < k:
        return None
    deadline = deadline if deadline is not None else time.monotonic() + 120
    t_scale = float(np.max(t_data))
    y_scale = max(float(np.max(np.abs(y_data))), np.finfo(float).eps)
    t_norm, y_norm = t_data / t_scale, y_data / y_scale
    scales = np.r_[y_scale, y_scale, np.ones(n), np.full(n, y_scale / t_scale), np.full(n, t_scale)]
    fixed = {2 + n - 1: 0.0}  # Remove softmax translation invariance.
    if force_yi:
        fixed[0] = 0.0
    if use_first_order_phase1:
        fixed[2 + 2 * n] = 0.0
    free = np.array([i for i in range(2 + 3 * n) if i not in fixed])
    all_bounds = (
        [(0.0, 1.5), (0.0, 2.0)] + [(-20.0, 20.0)] * n + [(0.0, 500.0)] * n + [(0.0, 1.2)] * n
    )
    bounds = [all_bounds[i] for i in free]
    lower, upper = np.array(bounds).T
    guess = smart_initial_guess(t_data, y_data, n) / scales
    guess[2 : 2 + n] = 0.0
    template = np.zeros(2 + 3 * n)
    for i, value in fixed.items():
        template[i] = value

    def expand(values):
        theta = template.copy()
        theta[free] = values
        return theta

    def predict(values):
        return evaluate_polyauxic_model(
            t_norm, expand(values), model_func, n, use_first_order_phase1
        )

    calls = 0

    def loss(values):
        nonlocal calls
        calls += 1
        if calls % 64 == 1 and time.monotonic() > deadline:
            raise AnalysisTimeout("Analysis time limit reached")
        theta = expand(values)
        prediction = predict(values)
        residual = y_norm - prediction
        # Chronological lags, positive amplitude and the original lower-response penalty.
        lag = theta[2 + 2 * n :]
        penalty = 1e6 * np.sum(np.maximum(0.0, -np.diff(lag)) ** 2)
        penalty += 1e6 * max(0.0, theta[0] - theta[1] + 1e-9) ** 2
        penalty += 1e6 * np.sum(np.maximum(0.0, -0.1 * np.max(np.abs(y_norm)) - prediction) ** 2)
        base = np.sum(2 * (np.hypot(1.0, residual) - 1)) if robust else np.sum(residual**2)
        return float(base + penalty)

    rng = np.random.default_rng(random_seed)
    population = np.tile(guess[free], (50, 1)) * rng.uniform(0.8, 1.2, (50, len(free)))
    for j, index in enumerate(free):
        if 2 <= index < 2 + n:
            population[:, j] = rng.uniform(-2.0, 2.0, 50)
    population = np.clip(population, lower, upper)
    population[0] = np.clip(guess[free], lower, upper)
    de = differential_evolution(
        loss,
        bounds,
        maxiter=3000,
        init=population,
        seed=random_seed,
        polish=False,
        tol=1e-6,
        atol=1e-12,
    )
    local = minimize(loss, de.x, method="L-BFGS-B", bounds=bounds, tol=1e-10)
    chosen = (
        local
        if local.success
        and np.isfinite(local.fun)
        and local.fun <= de.fun + max(1e-12, abs(de.fun) * 1e-8)
        else de
    )
    theta_norm = expand(chosen.x)
    theta = theta_norm * scales
    prediction = evaluate_polyauxic_model(t_data, theta, model_func, n, use_first_order_phase1)
    if not np.all(np.isfinite(prediction)) or theta[1] <= theta[0]:
        return None
    if np.any(np.diff(theta[2 + 2 * n :]) < -1e-7 * t_scale):
        return None
    result = {
        "n_phases": n,
        "theta": theta,
        "y_pred": prediction,
        "use_first_order_phase1": bool(use_first_order_phase1),
        "converged": bool(chosen.success),
        "n_parameters": k,
        "optimizer_message": str(chosen.message),
        "random_seed": random_seed,
    }
    if robust:
        return result
    # Residual Jacobian avoids the old singular penalized Hessian and its missing
    # factor of two. Do not turn negative covariance diagonals into plausible SEs.
    jac = np.empty((len(t_data), len(free)))
    for j in range(len(free)):
        step = 1e-5 * max(1.0, abs(chosen.x[j]))
        high, low = chosen.x.copy(), chosen.x.copy()
        high[j] += step
        low[j] -= step
        jac[:, j] = (predict(high) - predict(low)) / (2 * step)
    se = np.full(len(theta), np.nan)
    for i in fixed:
        se[i] = 0.0
    se_p = np.full(n, np.nan)
    rank = int(np.linalg.matrix_rank(jac))
    at_bound = np.isclose(chosen.x, lower, rtol=0, atol=1e-7) | np.isclose(
        chosen.x, upper, rtol=0, atol=1e-7
    )
    reliable = rank == len(free) and not np.any(at_bound) and result["converged"]
    if reliable:
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
        variance = np.sum((y_norm - predict(chosen.x)) ** 2) / (len(y_norm) - k)
        covariance = variance * (vt.T / singular**2) @ vt
        se[free] = np.sqrt(np.maximum(0.0, np.diag(covariance))) * scales[free]
        full_cov = np.zeros((len(theta), len(theta)))
        full_cov[np.ix_(free, free)] = covariance
        se_p = calculate_p_errors(theta[2 : 2 + n], full_cov[2 : 2 + n, 2 : 2 + n])
    if n == 1:
        se_p[0] = 0.0
    if use_first_order_phase1:
        se[2 + 2 * n] = np.nan
    result.update(
        se=se,
        se_p=se_p,
        uncertainty_reliable=reliable,
        jacobian_rank=rank,
        metrics=calculate_metrics(y_data, prediction, k),
        outliers=np.zeros(len(y_data), dtype=bool),
    )
    return result


def fit_model_auto(
    t_data,
    y_data,
    model_func,
    n_phases,
    force_yi=False,
    force_yf=False,
    use_first_order_phase1=False,
    random_seed=42,
    deadline=None,
):
    return _fit(
        t_data,
        y_data,
        model_func,
        n_phases,
        force_yi=force_yi,
        force_yf=force_yf,
        use_first_order_phase1=use_first_order_phase1,
        random_seed=random_seed,
        deadline=deadline,
    )


def fit_model_auto_robust_pre(
    t_data,
    y_data,
    model_func,
    n_phases,
    force_yi=False,
    force_yf=False,
    use_first_order_phase1=False,
    random_seed=42,
    deadline=None,
):
    return _fit(
        t_data,
        y_data,
        model_func,
        n_phases,
        force_yi=force_yi,
        force_yf=force_yf,
        use_first_order_phase1=use_first_order_phase1,
        random_seed=random_seed,
        robust=True,
        deadline=deadline,
    )
