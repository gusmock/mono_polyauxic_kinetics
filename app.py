"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 

"""
"""
Polyauxic Robustness Simulator
------------------------------

This Streamlit app performs Monte Carlo robustness tests for the polyauxic
Boltzmann (Eq. 31) and Gompertz (Eq. 32) models described in your paper.

HIGH-LEVEL WORKFLOW
===================

1) USER INPUTS
   -----------
   The user specifies:
   - Model type: Boltzmann or Gompertz.
   - Number of phases (n_phases): 1 to 10.
   - Simulation parameters:
       * y_i, y_f (global initial and final values).
       * For each phase j = 1..n_phases:
           - p_j (relative weight / contribution of phase j).
           - r_max_j (maximum rate of phase j).
           - lambda_j (time at which phase j is centered / inflection).
     NOTE: For simulation, p_j are directly used and normalized internally
           so that sum_j p_j = 1. Fitting uses the softmax parametrization
           (z_j -> p_j) exactly as in the original implementation.

   - Minimum absolute deviation (abs_dev_min >= 0).
   - Maximum absolute deviation (abs_dev_max >= abs_dev_min).
   - Number of replicates: 1 to 5.
   - Number of points per replicate: 5 to 100.
   - Number of Monte Carlo tests: 1 to 100.

2) DATA GENERATION
   ----------------
   For a fixed set of "true" parameters and a chosen model:

   (a) Time grid:
       We construct a 1D time array t_sim with N_points equally spaced
       between:
           t_min = 0
           t_max = max( 3 * max(lambda_j), 1.0 )
       This is a heuristic that aims to cover the main dynamic range
       around the phase centers. If all lambda_j <= 0, we still use
       t_max = 1.0 to avoid degeneracy.

   (b) Noise-free signal:
       We compute the ideal, noise-free polyauxic curve
           y_true(t) = y_i + (y_f - y_i) * sum_j term_j(t)
       where term_j(t) is the Boltzmann (Eq. 31) or Gompertz (Eq. 32)
       term for phase j, built directly from the user-specified p_j,
       r_max_j, and lambda_j.

   (c) Noise model per data point:
       For each time point and for each replicate, the app generates
       a noisy observation y_obs via the Excel-like formula:

           y_obs = y_true
                    + ( abs_dev_min
                        + (abs_dev_max - abs_dev_min) * U_1 ) * N(0,1)

       where:
           - U_1 ~ Uniform(0,1),
           - N(0,1) is a standard normal random variable.
         In practice we implement this by:
           scale = abs_dev_min + (abs_dev_max - abs_dev_min) * rand()
           noise = scale * normal(0,1)

       Each replicate has independent noise. All replicates use the
       same time grid t_sim but different noise realizations.

   (d) Replicates:
       For each Monte Carlo test, we simulate `n_replicates` datasets,
       each of length N_points, and then flatten them into a single
       dataset {t_all, y_all} to be used for fitting.

3) FITTING
   -------
   We fit the same model (Boltzmann or Gompertz) with the same number
   of phases n_phases to the simulated data.

   - Parameterization in fitting:
       The fitting model uses:
           theta = [y_i, y_f, z_1..z_n, r_max_1..r_max_n, lambda_1..lambda_n]
       Internally we convert z_j into probabilities p_j via a softmax:
           p_j = exp(z_j) / sum_k exp(z_k)
       This matches the structure used in your main app.

   - Loss function:
       Sum of squared errors (SSE) between observed y and model y_pred.

   - Scaling:
       Time and response are normalized to improve conditioning:
           t_norm = t / max(t)
           y_norm = y / max(|y|)
       The optimization is performed in normalized space, then mapped
       back to the original scale.

   - Optimization:
       1) A differential evolution (DE) global search to escape poor
          local minima.
       2) A local L-BFGS-B refinement starting from the DE optimum.

       NOTE: For speed, this version uses a smaller population and
             fewer DE iterations than your original code. You can
             adjust `maxiter` and `popsize` if needed.

   - Metrics:
       For each Monte Carlo test we record:
         * All fitted parameters (y_i, y_f, p_j, r_max_j, lambda_j).
         * SSE.
         * R² and adjusted R².
         * AIC, AICc, BIC.

4) MONTE CARLO LOOP
   -----------------
   The simulation and fitting steps are repeated `n_tests` times.
   Each test uses a new random noise realization but the same true
   parameters and time grid.

   Results are stored in a pandas DataFrame with:
       - test_id
       - y_i_hat, y_f_hat
       - p1_hat..pn_hat
       - r_max1_hat..r_maxn_hat
       - lambda1_hat..lambdan_hat
       - SSE, R2, R2_adj, AIC, AICc, BIC

5) VISUALIZATION
   --------------
   The app displays:
   - A data table of all Monte Carlo results.
   - A multi-select line plot for any subset of parameter columns
     vs. test_id.
   - A line plot of AIC, AICc, BIC vs. test_id.
   - A line plot of R² and adjusted R² vs. test_id.

   This allows visual inspection of:
   - Parameter robustness (bias, variance, possible identifiability issues).
   - Stability of information criteria.
   - Fit quality consistency across repeated experiments.

USAGE SUMMARY (UI)
==================
1. Choose model ("Boltzmann" or "Gompertz").
2. Select number of phases (1–10).
3. Set y_i and y_f.
4. For each phase j, set p_j, r_max_j, lambda_j (defaults are provided).
5. Set min and max absolute deviation (noise level).
6. Set number of replicates, points per replicate, and number of tests.
7. Click "Run Monte Carlo Simulation".
8. Explore the results table and line plots.

All labels and UI text are in English; the response type is always the
generic y(x).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# 1. MODEL EQUATIONS (Eq. 31 and Eq. 32 terms)
# ---------------------------------------------------------------------------

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Single-phase Boltzmann term as in Eq. 31:

        y(x) = y_i + (y_f - y_i) * sum_j p_j / (1 + exp( ... ))

    Here we return ONLY the term:

        term_j(t) = p_j / (1 + exp( 4 * r_max_j * (lambda_j - t)
                                    / ((y_f - y_i) * p_j) + 2 ))

    The global y(t) is built by summing term_j(t) over j and then
    applying y_i + (y_f - y_i) * sum_terms.
    """
    t = np.asarray(t, dtype=float)
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
    Single-phase Gompertz term as in Eq. 32:

        y(x) = y_i + (y_f - y_i) * sum_j p_j * exp( -exp( ... ) )

    Here we return ONLY the term:

        term_j(t) = p_j * exp( -exp( (r_max_j * e * (lambda_j - t)
                                      / ((y_f - y_i) * p_j)) + 1 ) )

    Again, the global y(t) is:

        y(t) = y_i + (y_f - y_i) * sum_j term_j(t)
    """
    t = np.asarray(t, dtype=float)
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe * np.exp(-np.exp(exponent))


# ---------------------------------------------------------------------------
# 2. POLYAUXIC MODELS (SIMULATION vs. FITTING)
# ---------------------------------------------------------------------------

def polyauxic_simulation_model(t, y_i, y_f, p_vec, r_vec, lambda_vec, model_func):
    """
    Polyauxic model used for SIMULATION ONLY.

    Parameters
    ----------
    t : array-like
        Time vector.
    y_i, y_f : float
        Global initial and final values.
    p_vec : array-like of length n_phases
        Phase weights. Will be normalized internally to sum to 1.
    r_vec : array-like of length n_phases
        Phase maximum rates.
    lambda_vec : array-like of length n_phases
        Phase centers/inflection times.
    model_func : callable
        Either boltzmann_term_eq31 or gompertz_term_eq32.

    Returns
    -------
    y : ndarray
        Simulated response y(t).
    """
    t = np.asarray(t, dtype=float)
    p_vec = np.asarray(p_vec, dtype=float)
    r_vec = np.asarray(r_vec, dtype=float)
    lambda_vec = np.asarray(lambda_vec, dtype=float)
    n_phases = len(p_vec)

    # Normalize p_vec to sum to 1 (robust to arbitrary user input)
    p_sum = np.sum(p_vec)
    if p_sum <= 0:
        # Avoid degenerate case: equal weights if user provided all zeros
        p_vec = np.ones_like(p_vec) / max(n_phases, 1)
    else:
        p_vec = p_vec / p_sum

    sum_terms = np.zeros_like(t, dtype=float)
    for j in range(n_phases):
        sum_terms += model_func(t, y_i, y_f, p_vec[j], r_vec[j], lambda_vec[j])

    return y_i + (y_f - y_i) * sum_terms


def polyauxic_fit_model(t, theta, model_func, n_phases):
    """
    Polyauxic model used in FITTING.

    Parameters
    ----------
    t : array-like
        Time vector (normalized or real, depending on context).
    theta : array-like
        Parameter vector of length 2 + 3*n_phases:

            theta = [y_i, y_f,
                     z_1..z_n,
                     r_max_1..r_max_n,
                     lambda_1..lambda_n]

        z_j are internal parameters that are converted to probabilities
        p_j using a softmax transform, enforcing p_j >= 0 and sum p_j = 1.

    model_func : callable
        Boltzmann or Gompertz term function.
    n_phases : int
        Number of phases.

    Returns
    -------
    y : ndarray
        Model predictions y(t).
    """
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]

    # Softmax to convert z -> p
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_terms = np.zeros_like(t, dtype=float)
    for j in range(n_phases):
        sum_terms += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])

    return y_i + (y_f - y_i) * sum_terms


# ---------------------------------------------------------------------------
# 3. OPTIMIZATION: LOSS + INITIAL GUESS + FIT ENGINE
# ---------------------------------------------------------------------------

def sse_loss(theta, t, y, model_func, n_phases):
    """
    Sum of squared errors loss for normalized data.
    Penalizes solutions that produce heavily negative values.
    """
    y_pred = polyauxic_fit_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred) ** 2)


def smart_initial_guess(t, y, n_phases):
    """
    Rough initial guess using numerical derivative peaks.

    - Computes dy/dt.
    - Smooths dy/dt.
    - Detects peaks to locate candidate lambda_j and r_max_j.
    - Uses min(y) and max(y) as first guesses for y_i, y_f.

    This is a simplified version of your original heuristic.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    dy = np.gradient(y, t)
    # Simple moving average smoothing
    if len(dy) >= 5:
        dy_smooth = np.convolve(dy, np.ones(5) / 5, mode="same")
    else:
        dy_smooth = dy.copy()

    min_dist = max(1, len(t) // (n_phases * 4))
    if np.max(dy_smooth) > 0:
        peaks, props = find_peaks(
            dy_smooth,
            height=np.max(dy_smooth) * 0.1,
            distance=min_dist
        )
    else:
        peaks = np.array([], dtype=int)
        props = {"peak_heights": np.array([])}

    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props["peak_heights"])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({"lambda": t[p_idx], "r_max": abs(dy_smooth[p_idx])})
    # If fewer than n_phases peaks, fill with evenly spaced lambdas
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        if t_span <= 0:
            t_span = 1.0
        guesses.append(
            {
                "lambda": t.min() + t_span * (len(guesses) + 1) / (n_phases + 1),
                "r_max": (np.max(y) - np.min(y)) / (t_span / max(n_phases, 1)),
            }
        )

    guesses.sort(key=lambda x: x["lambda"])

    theta_guess = np.zeros(2 + 3 * n_phases)
    theta_guess[0] = np.min(y)
    theta_guess[1] = np.max(y)
    theta_guess[2 : 2 + n_phases] = 0.0  # z_j initial (softmax center)
    for i in range(n_phases):
        theta_guess[2 + n_phases + i] = guesses[i]["r_max"]
        theta_guess[2 + 2 * n_phases + i] = guesses[i]["lambda"]

    return theta_guess


def fit_model_auto(t_data, y_data, model_func, n_phases):
    """
    Main fitting function for one Monte Carlo realization.

    - Normalizes time and response.
    - Builds an initial guess via smart_initial_guess.
    - Runs differential evolution for global search.
    - Refines with L-BFGS-B.
    - Maps parameters back to real scale.
    - Computes R², adjusted R², SSE, AIC, AICc, BIC.

    Returns
    -------
    dict with keys:
        "theta"   : real-scale parameter vector (y_i, y_f, z_j, r_max_j, lambda_j)
        "metrics" : dict with R2, R2_adj, SSE, AIC, AICc, BIC
    or
        None if data are insufficient.
    """
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params:
        return None

    # Normalization to improve conditioning
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(np.abs(y_data)) if np.max(np.abs(y_data)) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale

    theta_smart = smart_initial_guess(t_data, y_data, n_phases)

    # Normalize initial guess
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2 : 2 + n_phases] = 0.0
    theta0_norm[2 + n_phases : 2 + 2 * n_phases] = (
        theta_smart[2 + n_phases : 2 + 2 * n_phases] * t_scale / y_scale
    )
    theta0_norm[2 + 2 * n_phases : 2 + 3 * n_phases] = (
        theta_smart[2 + 2 * n_phases : 2 + 3 * n_phases] / t_scale
    )

    # Differential evolution
    SEED_VALUE = 42
    np.random.seed(SEED_VALUE)
    pop_size = 20
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    bounds = []
    bounds.append((-0.2, 1.5))  # y_i_norm
    bounds.append((0.0, 2.0))   # y_f_norm
    for _ in range(n_phases):
        bounds.append((-10, 10))        # z
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))     # r_max_norm
    for _ in range(n_phases):
        bounds.append((-0.1, 1.2))      # lambda_norm

    res_de = differential_evolution(
        sse_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=800,        # reduced for speed; adjust if needed
        popsize=pop_size,
        init=init_pop,
        strategy="best1bin",
        seed=SEED_VALUE,
        polish=True,
        tol=1e-6,
    )

    res_opt = minimize(
        sse_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-10,
    )

    theta_norm = res_opt.x

    # Map back to real scale
    theta_real = np.zeros_like(theta_norm)
    theta_real[0] = theta_norm[0] * y_scale
    theta_real[1] = theta_norm[1] * y_scale
    theta_real[2 : 2 + n_phases] = theta_norm[2 : 2 + n_phases]
    theta_real[2 + n_phases : 2 + 2 * n_phases] = (
        theta_norm[2 + n_phases : 2 + 2 * n_phases] * (y_scale / t_scale)
    )
    theta_real[2 + 2 * n_phases : 2 + 3 * n_phases] = (
        theta_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * t_scale
    )

    # Predictions and metrics in REAL scale
    y_pred = polyauxic_fit_model(t_data, theta_real, model_func, n_phases)

    sse = np.sum((y_data - y_pred) ** 2)
    sst = np.sum((y_data - np.mean(y_data)) ** 2)
    if sse <= 1e-12:
        sse = 1e-12

    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    n_len = len(y_data)
    k = len(theta_real)
    if (n_len - k - 1) > 0:
        r2_adj = 1.0 - (1.0 - r2) * (n_len - 1) / (n_len - k - 1)
    else:
        r2_adj = np.nan

    aic = n_len * np.log(sse / n_len) + 2 * k
    bic = n_len * np.log(sse / n_len) + k * np.log(n_len)
    aicc = (
        aic + (2 * k * (k + 1)) / (n_len - k - 1)
        if (n_len - k - 1) > 0
        else np.inf
    )

    metrics = {
        "R2": r2,
        "R2_adj": r2_adj,
        "SSE": sse,
        "AIC": aic,
        "AICc": aicc,
        "BIC": bic,
    }

    return {"theta": theta_real, "metrics": metrics}


# ---------------------------------------------------------------------------
# 4. MONTE CARLO SIMULATION LOOP
# ---------------------------------------------------------------------------

def run_monte_carlo(
    model_func,
    n_phases,
    y_i_true,
    y_f_true,
    p_true,
    r_true,
    lambda_true,
    abs_dev_min,
    abs_dev_max,
    n_replicates,
    n_points,
    n_tests,
):
    """
    Executes the full Monte Carlo experiment:
        - Builds time grid.
        - Generates noise-free y(t).
        - For each test:
            * Adds noise for each replicate.
            * Flattens all replicates.
            * Fits the model.
            * Stores fitted parameters and metrics.

    Returns
    -------
    DataFrame with one row per test.
    """
    p_true = np.asarray(p_true, dtype=float)
    r_true = np.asarray(r_true, dtype=float)
    lambda_true = np.asarray(lambda_true, dtype=float)

    # Time grid heuristic
    max_lambda = np.max(lambda_true) if len(lambda_true) > 0 else 1.0
    if np.isfinite(max_lambda) and max_lambda > 0:
        t_max = 3.0 * max_lambda
    else:
        t_max = 1.0
    t_min = 0.0
    t_sim = np.linspace(t_min, t_max, n_points)

    # Noise-free signal
    y_true = polyauxic_simulation_model(
        t_sim,
        y_i_true,
        y_f_true,
        p_true,
        r_true,
        lambda_true,
        model_func,
    )

    results = []

    for test_id in range(1, n_tests + 1):
        all_t = []
        all_y = []

        for _ in range(n_replicates):
            # Random scales per point
            u_uniform = np.random.rand(len(t_sim))
            scales = abs_dev_min + (abs_dev_max - abs_dev_min) * u_uniform
            noise = scales * np.random.normal(loc=0.0, scale=1.0, size=len(t_sim))
            y_obs = y_true + noise

            all_t.append(t_sim)
            all_y.append(y_obs)

        t_all = np.concatenate(all_t)
        y_all = np.concatenate(all_y)

        fit_res = fit_model_auto(t_all, y_all, model_func, n_phases)
        if fit_res is None:
            # Insufficient data or error in fitting
            continue

        theta_hat = fit_res["theta"]
        metrics = fit_res["metrics"]

        # Decode parameters
        y_i_hat = theta_hat[0]
        y_f_hat = theta_hat[1]
        z_hat = theta_hat[2 : 2 + n_phases]
        r_hat = theta_hat[2 + n_phases : 2 + 2 * n_phases]
        lambda_hat = theta_hat[2 + 2 * n_phases : 2 + 3 * n_phases]

        # Softmax -> p_hat
        z_shift = z_hat - np.max(z_hat)
        exp_z = np.exp(z_shift)
        p_hat = exp_z / np.sum(exp_z)

        row = {
            "test_id": test_id,
            "y_i_hat": y_i_hat,
            "y_f_hat": y_f_hat,
            "SSE": metrics["SSE"],
            "R2": metrics["R2"],
            "R2_adj": metrics["R2_adj"],
            "AIC": metrics["AIC"],
            "AICc": metrics["AICc"],
            "BIC": metrics["BIC"],
        }

        # Phase parameters
        for j in range(n_phases):
            row[f"p{j+1}_hat"] = p_hat[j]
            row[f"r_max{j+1}_hat"] = r_hat[j]
            row[f"lambda{j+1}_hat"] = lambda_hat[j]

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Polyauxic Robustness Simulator",
        layout="wide",
    )

    st.title("Polyauxic Robustness Simulator")
    st.markdown(
        "Monte Carlo robustness analysis for polyauxic Boltzmann and Gompertz models."
    )

    st.sidebar.header("Simulation Settings")

    # Choose model
    model_name = st.sidebar.selectbox(
        "Model",
        ["Boltzmann (Eq. 31)", "Gompertz (Eq. 32)"],
    )
    if "Boltzmann" in model_name:
        model_func = boltzmann_term_eq31
    else:
        model_func = gompertz_term_eq32

    # Number of phases
    n_phases = st.sidebar.number_input(
        "Number of phases",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    st.sidebar.subheader("Global Parameters")
    y_i_true = st.sidebar.number_input("y_i (initial value)", value=0.0)
    y_f_true = st.sidebar.number_input("y_f (final value)", value=1.0)

    # Phase-specific parameters
    p_true = []
    r_true = []
    lambda_true = []

    st.sidebar.subheader("Phase Parameters")
    for j in range(1, n_phases + 1):
        with st.sidebar.expander(f"Phase {j}", expanded=(j <= 2)):
            default_p = 1.0 / n_phases
            p_j = st.number_input(
                f"p_{j}",
                min_value=0.0,
                value=float(default_p),
                key=f"p_{j}",
            )
            r_j = st.number_input(
                f"r_max_{j}",
                value=1.0,
                key=f"r_{j}",
            )
            lambda_j = st.number_input(
                f"lambda_{j}",
                value=float(j),
                key=f"lambda_{j}",
            )
            p_true.append(p_j)
            r_true.append(r_j)
            lambda_true.append(lambda_j)

    st.sidebar.subheader("Noise Settings")
    abs_dev_min = st.sidebar.number_input(
        "Minimum absolute deviation",
        min_value=0.0,
        value=0.0,
    )
    abs_dev_max = st.sidebar.number_input(
        "Maximum absolute deviation",
        min_value=0.0,
        value=0.1,
    )

    if abs_dev_max < abs_dev_min:
        st.sidebar.error("Maximum deviation must be >= minimum deviation.")

    st.sidebar.subheader("Monte Carlo Design")
    n_replicates = st.sidebar.number_input(
        "Number of replicates",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )
    n_points = st.sidebar.number_input(
        "Points per replicate",
        min_value=5,
        max_value=100,
        value=30,
        step=1,
    )
    n_tests = st.sidebar.number_input(
        "Number of Monte Carlo tests",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )

    run_button = st.sidebar.button("Run Monte Carlo Simulation")

    if run_button:
        if abs_dev_max < abs_dev_min:
            st.error("Fix noise settings: maximum deviation must be >= minimum.")
            return

        st.info("Running Monte Carlo simulation...")

        df_results = run_monte_carlo(
            model_func=model_func,
            n_phases=n_phases,
            y_i_true=y_i_true,
            y_f_true=y_f_true,
            p_true=p_true,
            r_true=r_true,
            lambda_true=lambda_true,
            abs_dev_min=abs_dev_min,
            abs_dev_max=abs_dev_max,
            n_replicates=n_replicates,
            n_points=n_points,
            n_tests=n_tests,
        )

        if df_results.empty:
            st.error("No results. Possibly insufficient data or optimization failure.")
            return

        st.success("Simulation completed.")

        st.subheader("Monte Carlo Results (per test)")
        st.dataframe(df_results)

        # Download CSV
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="polyauxic_robustness_results.csv",
            mime="text/csv",
        )

        # Parameter columns
        param_cols = ["y_i_hat", "y_f_hat"]
        for j in range(1, n_phases + 1):
            param_cols.append(f"p{j+0}_hat")
        for j in range(1, n_phases + 1):
            param_cols.append(f"r_max{j}_hat")
        for j in range(1, n_phases + 1):
            param_cols.append(f"lambda{j}_hat")

        st.markdown("---")
        st.subheader("Parameter behavior across Monte Carlo tests")

        selected_params = st.multiselect(
            "Select parameters to plot",
            options=param_cols,
            default=["y_i_hat", "y_f_hat"],
        )

        if selected_params:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in selected_params:
                ax.plot(df_results["test_id"], df_results[col], marker="o", label=col)
            ax.set_xlabel("Test ID")
            ax.set_ylabel("Estimated value")
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Information criteria vs. test")

        fig_ic, ax_ic = plt.subplots(figsize=(10, 5))
        for col in ["AIC", "AICc", "BIC"]:
            ax_ic.plot(
                df_results["test_id"],
                df_results[col],
                marker="o",
                label=col,
            )
        ax_ic.set_xlabel("Test ID")
        ax_ic.set_ylabel("Criterion value")
        ax_ic.grid(True, linestyle=":", alpha=0.3)
        ax_ic.legend()
        st.pyplot(fig_ic)

        st.markdown("---")
        st.subheader("R² and Adjusted R² vs. test")

        fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))
        ax_r2.plot(
            df_results["test_id"],
            df_results["R2"],
            marker="o",
            label="R²",
        )
        ax_r2.plot(
            df_results["test_id"],
            df_results["R2_adj"],
            marker="s",
            label="Adjusted R²",
        )
        ax_r2.set_xlabel("Test ID")
        ax_r2.set_ylabel("Value")
        ax_r2.grid(True, linestyle=":", alpha=0.3)
        ax_r2.legend()
        st.pyplot(fig_r2)


if __name__ == "__main__":
    main()
