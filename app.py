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
================================================================================
POLYAUXIC ROBUSTNESS SIMULATOR (v3.2 - ROUT & Visualization)
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)
GitHub: https://github.com/gusmock/mono_polyauxic_kinetics/

--------------------------------------------------------------------------------
HIGH-LEVEL OVERVIEW
--------------------------------------------------------------------------------
This Streamlit application performs Monte Carlo robustness analysis of polyauxic
( multi-phase ) kinetic models under controlled noise and outlier scenarios.

Key features:
- Flexible polyauxic growth curves based on Boltzmann or Gompertz equations.
- Automatic experimental design enforcing a 10% plateau rule.
- Global + local nonlinear fitting with constrained parameterization.
- Outlier detection via:
  * Simple MAD-based rule (ROUT-like),
  * A ROUT-style procedure combining robust regression and FDR control
    (Motulsky & Brown, 2006; Benjamini & Hochberg, 1995).

The app is intended for rigorous in silico validation of polyauxic models
and for quantifying how parameter estimates behave under noise and outliers.

--------------------------------------------------------------------------------
PIPELINE / CODE ORGANIZATION
--------------------------------------------------------------------------------

1. MODEL DEFINITION (Section 1 & 2)
   - Two base "single-phase" kernels are implemented:
     * boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j)
       → symmetric sigmoidal.
     * gompertz_term(t, y_i, y_f, p_j, r_j, lam_j)
       → asymmetric, classical Gompertz-type.
   - A polyauxic model is constructed as a sum of J phases:
       y(t) = y_i + (y_f - y_i) * Σ_j f_j(t; p_j, r_j, λ_j)
     where each f_j is either Boltzmann or Gompertz.
   - Proportions p_j represent the fractional contribution of each phase.
     In the fitting engine, p_j are parametrized via a softmax transform,
     ensuring Σ_j p_j = 1 by construction.
   - Lag times λ_j are automatically sorted so that λ_1 < λ_2 < ... < λ_J,
     enforcing a physically consistent sequence of phases.

2. AUTOMATIC SATURATION TIME & EXPERIMENTAL DESIGN (find_saturation_time)
   - For a given parameter set (y_i, y_f, p, r, λ) and chosen kernel:
     * A target normalized value (0.99) is used to determine t_99 such that
       y(t_99) ≈ 0.99 * y_f.
     * Total experimental time t_total is defined as:
           t_total = t_99 / 0.90
       so that approximately 90% of the sampled points lie in the growth
       region and 10% in the plateau, implementing the "10% plateau rule".
   - This logic is used to generate the simulation time vector t_sim and to
     show an experimental design preview to the user.

3. DATA GENERATION (polyauxic_components)
   - Using t_sim, the global parameters (y_i, y_f) and per-phase parameters
     (p, r, λ), the code computes:
     * y_total(t): the full polyauxic curve,
     * components_list: each phase contribution for visualization.
   - These deterministic curves are the "true" underlying behavior for
     the Monte Carlo experiments.

4. MONTE CARLO ENGINE (monte_carlo & monte_carlo_single)
   For each Monte Carlo "test":
   - Generate n_rep replicate time series with n_points each.
   - For each replicate:
     * Draw a per-point noise scale σ_i uniformly between dev_min and dev_max.
     * Apply additional "volatility" to mimic heteroscedastic noise.
     * Randomly mark a fraction outlier_pct of points as "outliers" and
       inflate their noise scale by a factor between 3 and 6.
     * Generate noisy observations:
           y_obs = y_true + ε
       where ε ~ N(0, σ_i^2), with σ_i larger for outlier points.
   - Concatenate all replicate data into single arrays t_all, y_all.
   - Apply the chosen outlier strategy (None, simple MAD, or ROUT), then
     perform nonlinear regression and record fitted parameters and metrics.

5. FITTING ENGINE & PARAMETERIZATION (Section 3)
   - Parameters are stored in a flat θ vector and unpacked via:
       unpack_parameters(theta, n_phases)
     where:
       * y_i = θ[0], y_f = θ[1],
       * z_j = θ[2 : 2+n_phases] → softmax(z_j) gives p_j,
       * r_j = |θ[...]| ensures positive rates,
       * λ_j are extracted and then sorted.
   - Two loss functions are available:
     * sse_loss: classic sum of squared residuals (least squares).
     * robust_loss: Soft L1 loss (M-estimator style) to down-weight outliers.
   - fit_polyauxic:
     * Normalizes time and response amplitude to improve conditioning.
     * Uses smart_guess to initialize λ_j and r_j based on derivative peaks.
     * Runs differential_evolution (global search) followed by minimize
       (L-BFGS-B local refinement) under given bounds.
     * Rescales fitted parameters back to original units.
     * Computes goodness-of-fit metrics: SSE, R², adjusted R², AIC, AICc, BIC.

6. OUTLIER DETECTION STRATEGIES (Section 0 & in monte_carlo_single)
   - "None":
     * Single SSE-based fit, no outlier removal.
   - "ROUT-like (Simple MAD)":
     * Step 1: Fit using SSE.
     * Step 2: Compute residuals and a robust scale estimate via MAD, then
       flag points with |residual|/σ_MAD > 2.5 as outliers.
     * Step 3: Refit using SSE on the cleaned dataset if enough points remain;
       otherwise keep the full dataset.
   - "ROUT (Robust + FDR)" (UPDATED IMPLEMENTATION):
     * Step 1: Robust fit using the Soft L1 loss to obtain a baseline model
       that is less influenced by outliers, as in Motulsky & Brown (2006).
     * Step 2: Compute residuals and a robust standard deviation (RSDR) from
       the median absolute deviation (MAD).
     * Step 3: Normalize residuals into t-like statistics and compute
       two-tailed p-values via the Student t-distribution.
     * Step 4: Apply the Benjamini–Hochberg FDR procedure at level Q%:
       - Sort p-values from smallest to largest.
       - Find the largest index k such that
             p_(k) ≤ (k / n) * (Q / 100),
         where n is the number of data points.
       - All points with p ≤ p_(k) are declared outliers.
     * Step 5: Refit using SSE on the remaining (non-outlier) data, if there
       are enough points; otherwise, revert to fitting the full dataset.

7. RESULT AGGREGATION AND VISUALIZATION (Section 5 & Execution)
   - After n_tests Monte Carlo runs:
     * Compute mean and standard deviation of simulated y over all tests.
     * Plot:
       - True generating curve,
       - Simulation mean ± standard deviation,
       - Detected outliers.
     * Display overall counts of simulated points and removed outliers.
     * Show per-test trajectories of R², adjusted R², AIC, AICc, BIC.
     * Display boxplots of the distributions of fitted parameters (y_i, y_f,
       p_j, r_j, λ_j).
     * Plot per-test parameter estimates vs. true values for bias and
       stability analysis.
     * Provide CSV export of all Monte Carlo results.

--------------------------------------------------------------------------------
MAIN REFERENCES IMPLEMENTED IN THIS CODE
--------------------------------------------------------------------------------
1. Motulsky, H. J., & Brown, R. E. (2006).
   Detecting outliers when fitting data with nonlinear regression – a new
   method based on robust nonlinear regression and the false discovery rate.
   BMC Bioinformatics, 7, 123.

2. Benjamini, Y., & Hochberg, Y. (1995).
   Controlling the False Discovery Rate: A Practical and Powerful Approach
   to Multiple Testing.
   Journal of the Royal Statistical Society. Series B (Methodological),
   57(1), 289–300.

3. Rousseeuw, P. J., & Croux, P. J. (1993).
   Alternatives to the Median Absolute Deviation.
   Journal of the American Statistical Association, 88(424), 1273–1283.

4. Numerical implementation builds on:
   - SciPy (optimization, root finding, statistics),
   - NumPy (array operations),
   - Pandas (tabular aggregation),
   - Matplotlib (plotting),
   - Streamlit (interactive UI).

"""

import streamlit as st
import streamlit.components.v1 as st_components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, brentq
from scipy.signal import find_peaks
from scipy.stats import t as t_dist 
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Polyauxic Robustness Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 0. HELPER FUNCTIONS: OUTLIER DETECTION STRATEGIES
# ------------------------------------------------------------

def detect_outliers_mad_simple(y_true, y_pred):
    """
    ROUT-like (Simple): Uses residuals from Least Squares fit.
    Threshold: Fixed Z-score > 2.5 based on MAD.
    """
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5


def detect_outliers_rout_rigorous(y_true, y_pred, Q=1.0):
    """
    ROUT (Rigorous, Motulsky & Brown, 2006 style):
    - Assume y_pred comes from a robust fit (e.g., Soft L1 regression).
    - Compute robust residual scale via MAD.
    - Convert residuals into t-like statistics.
    - Compute two-tailed p-values using a t-distribution.
    - Apply Benjamini–Hochberg FDR procedure at level Q (%).
    
    Parameters
    ----------
    y_true : array-like
        Observed data.
    y_pred : array-like
        Model predictions from a robust fit.
    Q : float, optional
        Desired maximum False Discovery Rate in percent (e.g., 1.0 for 1% FDR).
    
    Returns
    -------
    mask_outliers : np.ndarray of bool
        Boolean array where True indicates an outlier according to ROUT.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n = residuals.size

    if n < 3:
        # Not enough points to perform any meaningful FDR-based test
        return np.zeros_like(residuals, dtype=bool)

    # Robust standard deviation (RSDR) via MAD of residuals
    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    rsdr = 1.4826 * mad_res if mad_res > 1e-12 else 1e-12

    # t-like statistics (normalized residuals)
    t_scores = residuals / rsdr

    # Two-tailed p-values using Student's t-distribution
    # df is approximated as n-1 here (in full nonlinear regression, df would be n - p)
    df = max(n - 1, 1)
    abs_t = np.abs(t_scores)
    p_values = 2.0 * (1.0 - t_dist.cdf(abs_t, df=df))

    # Benjamini–Hochberg FDR control at level Q%
    alpha = Q / 100.0
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]

    # Compute BH thresholds: (i/n) * alpha
    i = np.arange(1, n + 1)
    bh_thresholds = (i / n) * alpha

    # Find largest k where p_(k) <= (k/n)*alpha
    below = p_sorted <= bh_thresholds
    if not np.any(below):
        # No significant outliers at this FDR level
        return np.zeros_like(residuals, dtype=bool)

    k_max = np.max(np.where(below)[0])  # index in sorted array (0-based)
    p_crit = p_sorted[k_max]

    # Mark as outliers all points with p <= p_crit
    mask_outliers = p_values <= p_crit
    return mask_outliers

# ------------------------------------------------------------
# 1. MODEL EQUATIONS
# ------------------------------------------------------------
def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i 
    if dy < 1e-12: dy = 1e-12
    
    p = max(p_j, 1e-12)
    numerator = 4 * r_j * (lam_j - t)
    denominator = dy * p
    expo = numerator / denominator + 2
    expo = np.clip(expo, -500, 500) 
    return p / (1 + np.exp(expo))

def gompertz_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i 
    if dy < 1e-12: dy = 1e-12
    
    p = max(p_j, 1e-12)
    numerator = r_j * np.e * (lam_j - t)
    denominator = dy * p
    expo = numerator / denominator + 1
    expo = np.clip(expo, -500, 500)
    return p * np.exp(-np.exp(expo))

# ------------------------------------------------------------
# 2. GENERATING FUNCTIONS
# ------------------------------------------------------------
def polyauxic_func_normalized(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    sum_terms = 0.0
    for j in range(len(p_vec)):
        sum_terms += func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
    return sum_terms

def polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    idx = np.argsort(lam_vec)
    lam_vec = np.array(lam_vec)[idx]
    p_vec = np.array(p_vec)[idx]
    r_vec = np.array(r_vec)[idx]
    
    components_list = []
    sum_terms = np.zeros_like(t, dtype=float)
    
    for j in range(len(p_vec)):
        term = func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
        components_list.append((y_f - y_i) * term)
        sum_terms += term
        
    y_total = y_i + (y_f - y_i) * sum_terms
    return y_total, components_list

def find_saturation_time(y_i, y_f, p_vec, r_vec, lam_vec, func):
    target = 0.99
    def objective(t):
        return polyauxic_func_normalized(t, y_i, y_f, p_vec, r_vec, lam_vec, func) - target

    t_start = max(lam_vec)
    t_end = t_start * 2 + 50 
    iter_limit = 0
    while objective(t_end) < 0 and iter_limit < 50:
        t_end *= 1.5
        iter_limit += 1
    
    try:
        t_99 = brentq(objective, 0, t_end)
    except:
        t_99 = t_end 

    t_total = t_99 / 0.90
    return t_total, t_99

# ------------------------------------------------------------
# 3. FITTING ENGINE & LOSS FUNCTIONS
# ------------------------------------------------------------

def unpack_parameters(theta, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    p_raw = expz / np.sum(expz)
    r_raw = np.abs(theta[2 + n_phases : 2 + 2*n_phases])
    lam_raw = theta[2 + 2*n_phases : 2 + 3*n_phases]
    
    # Sort triplets by time
    sort_idx = np.argsort(lam_raw)
    lam = lam_raw[sort_idx]
    p = p_raw[sort_idx]
    r = r_raw[sort_idx]
    return y_i, y_f, p, r, lam

def polyauxic_fit_model(t, theta, func, n_phases):
    y_i, y_f, p, r, lam = unpack_parameters(theta, n_phases)
    sum_terms = np.zeros_like(t)
    for j in range(n_phases):
        sum_terms += func(t, y_i, y_f, p[j], r[j], lam[j])
    return y_i + (y_f - y_i) * sum_terms

# --- LOSS FUNCTION 1: SSE (Least Squares) ---
def sse_loss(theta, t, y, func, n_phases):
    if theta[0] >= theta[1]: return 1e12 
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

# --- LOSS FUNCTION 2: SOFT L1 (Robust) ---
def robust_loss(theta, t, y, func, n_phases):
    if theta[0] >= theta[1]: return 1e12 
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): return 1e12
    residuals = y - y_pred
    loss = 2 * (np.sqrt(1 + residuals**2) - 1)
    return np.sum(loss)

def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    dy_s = np.convolve(dy, np.ones(5)/5, mode='same') if len(dy)>=5 else dy
    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guesses = []
    if len(peaks) > 0:
        idx = np.argsort(props['peak_heights'])[::-1][:n_phases]
        best = peaks[idx]
        for p in best: guesses.append((t[p], abs(dy_s[p])))
    while len(guesses) < n_phases:
        tspan = t.max() - t.min() if t.max() > t.min() else 1
        guesses.append((t.min() + tspan*(len(guesses)+1)/(n_phases+1), (y.max()-y.min())/(tspan/n_phases)))
    
    guesses = sorted(guesses, key=lambda x: x[0])
    theta0 = np.zeros(2 + 3*n_phases)
    theta0[0] = max(0, y.min()) 
    theta0[1] = max(y.max(), y.min() + 0.1)
    for i,(lam_guess, r_guess) in enumerate(guesses):
        theta0[2 + n_phases + i] = r_guess
        theta0[2 + 2*n_phases + i] = lam_guess
    return theta0

def fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss):
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1
    
    t_n = t_all / t_scale
    y_n = y_all / y_scale
    
    theta0 = smart_guess(t_all, y_all, n_phases)
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale
    th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0 
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases] * (t_scale/y_scale) 
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases] / t_scale
    
    bounds = [(0.0, 1.5), (0.001, 2.0)] + [(-10,10)]*n_phases + [(0,500)]*n_phases + [(-0.2,1.2)]*n_phases
    
    popsize = 20
    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))
    
    res_de = differential_evolution(
        loss_function, bounds, args=(t_n, y_n, func, n_phases),
        maxiter=1000, popsize=popsize, init=init_pop, 
        strategy="best1bin", polish=True, tol=1e-6
    )
    
    res = minimize(
        loss_function, res_de.x, args=(t_n, y_n, func, n_phases), 
        method="L-BFGS-B", bounds=bounds, tol=1e-10
    )
    
    th_n = res.x
    th = np.zeros_like(th_n)
    th[0] = th_n[0] * y_scale
    th[1] = th_n[1] * y_scale
    th[2:2+n_phases] = th_n[2:2+n_phases]
    th[2+n_phases:2+2*n_phases] = th_n[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    th[2+2*n_phases:2+3*n_phases] = th_n[2+2*n_phases:2+3*n_phases] * t_scale
    
    y_pred = polyauxic_fit_model(t_all, th, func, n_phases)
    sse = np.sum((y_all - y_pred)**2)
    sst = np.sum((y_all - np.mean(y_all))**2)
    r2 = 1 - sse/sst if sst > 1e-12 else 0
    n = len(y_all); k = len(th)
    r2adj = 1 - (1-r2)*(n-1)/(n-k-1) if (n-k-1)>0 else 0
    
    if sse > 1e-12:
        aic = n*np.log(sse/n) + 2*k
        bic = n*np.log(sse/n) + k*np.log(n)
        aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf
    else:
        aic, bic, aicc = -np.inf, -np.inf, -np.inf

    return th, {"SSE":sse,"R2":r2,"R2_adj":r2adj,"AIC":aic,"AICc":aicc,"BIC":bic}

# ------------------------------------------------------------
# 4. MONTE CARLO ENGINE
# ------------------------------------------------------------
def monte_carlo_single(test_idx, func, ygen, t_sim, p_true, r_true, lam_true,
                       dev_min, dev_max, n_rep, n_points, y_i, y_f, 
                       outlier_method, outlier_pct, rout_q):
    
    n_phases = len(p_true)
    t_all_list = []; y_all_list = []
    
    # ------------------ NOISE GENERATION ------------------
    n_outliers_gen = int(n_points * (outlier_pct / 100.0))
    
    for rep in range(n_rep):
        base_scales = dev_min + (dev_max - dev_min) * np.random.rand(n_points)
        volatility = np.random.uniform(0.8, 1.2, n_points)
        
        if n_outliers_gen > 0:
            bad_indices = np.random.choice(n_points, n_outliers_gen, replace=False)
            volatility[bad_indices] = np.random.uniform(3.0, 6.0, n_outliers_gen)
        
        final_scales = base_scales * volatility
        noise = final_scales * np.random.normal(0, 1, n_points)
        
        y_obs = ygen + noise
        t_all_list.append(t_sim); y_all_list.append(y_obs)
        
    t_all = np.concatenate(t_all_list)
    y_all = np.concatenate(y_all_list)
    
    # ------------------ FITTING & OUTLIER REMOVAL ------------------
    detected_outliers_t = []
    detected_outliers_y = []
    
    if outlier_method == "None":
        th, met = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        
    elif outlier_method == "ROUT-like (Simple MAD)":
        # 1. Fit (SSE) -> 2. Detect (MAD Z>2.5) -> 3. Refit (SSE)
        th_pre, _ = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        y_pred_pre = polyauxic_fit_model(t_all, th_pre, func, n_phases)
        
        mask = detect_outliers_mad_simple(y_all, y_pred_pre) # Returns True for outliers
        
        if np.any(mask):
            detected_outliers_t = t_all[mask]
            detected_outliers_y = y_all[mask]
            
        t_clean = t_all[~mask]; y_clean = y_all[~mask]
        
        if len(y_clean) < len(th_pre) + 5: 
             th, met = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        else:
             th, met = fit_polyauxic(t_clean, y_clean, func, n_phases, loss_function=sse_loss)
             
    elif outlier_method == "ROUT (Robust + FDR)":
        # 1. Fit (Robust) -> 2. Detect (ROUT Q) via BH-FDR -> 3. Refit (SSE)
        th_pre, _ = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=robust_loss)
        y_pred_pre = polyauxic_fit_model(t_all, th_pre, func, n_phases)
        
        mask = detect_outliers_rout_rigorous(y_all, y_pred_pre, Q=rout_q)
        
        if np.any(mask):
            detected_outliers_t = t_all[mask]
            detected_outliers_y = y_all[mask]
            
        t_clean = t_all[~mask]; y_clean = y_all[~mask]
        
        if len(y_clean) < len(th_pre) + 5: 
             th, met = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        else:
             th, met = fit_polyauxic(t_clean, y_clean, func, n_phases, loss_function=sse_loss)
    
    # ------------------ OUTPUT ------------------
    yi_hat, yf_hat, p_hat, r_hat, lam_hat = unpack_parameters(th, n_phases)
    
    row = {
        "test": test_idx, 
        "yi_hat": yi_hat, "yf_hat": yf_hat,
        "SSE": met["SSE"], "R2": met["R2"], "R2_adj": met["R2_adj"],
        "AIC": met["AIC"], "AICc": met["AICc"], "BIC": met["BIC"]
    }
    
    for j in range(n_phases):
        row[f"p{j+1}"] = p_hat[j]
        row[f"r{j+1}"] = r_hat[j]
        row[f"lam{j+1}"] = lam_hat[j]
        
    return row, t_all, y_all, detected_outliers_t, detected_outliers_y

def monte_carlo(func, ygen, t_sim, p_true, r_true, lam_true,
                dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, 
                outlier_method, outlier_pct, rout_q):
    results = []
    
    # Storage for global plotting
    all_out_t = []
    all_out_y = []
    y_matrix_stack = []

    progress = st.progress(0.0)
    status_text = st.empty()
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(monte_carlo_single, i+1, func, ygen, t_sim, p_true, r_true, lam_true,
                            dev_min, dev_max, n_rep, n_points, y_i, y_f, 
                            outlier_method, outlier_pct, rout_q): i+1 
            for i in range(n_tests)
        }
        done = 0
        for fut in as_completed(futures):
            row, t_dat, y_dat, out_t, out_y = fut.result()
            results.append(row)
            
            y_mat_local = y_dat.reshape(n_rep, n_points)
            y_matrix_stack.append(y_mat_local)
            
            if len(out_t) > 0:
                all_out_t.append(out_t)
                all_out_y.append(out_y)
                
            done += 1
            progress.progress(done / n_tests)
            status_text.text(f"Simulating Test {done}/{n_tests}...")
            
    status_text.text("Simulation finished successfully.")
    df = pd.DataFrame(results).sort_values("test")
    
    final_y_matrix = np.vstack(y_matrix_stack)
    y_mean = np.mean(final_y_matrix, axis=0)
    y_std = np.std(final_y_matrix, axis=0)
    
    if len(all_out_t) > 0:
        flat_out_t = np.concatenate(all_out_t)
        flat_out_y = np.concatenate(all_out_y)
    else:
        flat_out_t = np.array([])
        flat_out_y = np.array([])
        
    return df, y_mean, y_std, flat_out_t, flat_out_y, final_y_matrix.size

# ------------------------------------------------------------
# 5. STREAMLIT APP - UI
# ------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>Polyauxic Robustness Simulator</h1>", unsafe_allow_html=True)

with st.expander("📘 Instruction Manual & Parameter Rules (ROUT Enabled)"):
    st.markdown("""
    This simulator validates Polyauxic Kinetic Models under strict experimental design constraints.
    
    ### ⏱️ Experimental Design Logic (10% Plateau Rule)
    * **Automated Time Calculation:** Calculates Saturation Time ($t_{sat}$) at **99%** of $y_f$.
    * Time set so $t_{sat}$ = **90%** of data points (10% plateau).
    
    ### ⚖️ Parameter Constraints (Strictly Enforced)
    * **Proportions ($p$):** Sum is forced to **1.0**.
    * **Growth:** $y_i < y_f$.
    * **Sequence:** Lag times ($\lambda$) are mathematically **sorted triplets**. 
    * **Correction:** The fitting engine now correctly links $p$ and $r$ to their sorted $\lambda$.
    
    ### 🧪 Outlier & ROUT Logic
    * **Outlier %:** Injects artificial contaminants (3x-6x noise) into the data.
    * **ROUT-like (Simple):** Uses Standard LS Fit + MAD check (Threshold > 2.5 sigma).
    * **ROUT (Rigorous):** Uses Robust Fit (Soft L1) + MAD check + False Discovery Rate (Q) threshold.
    """)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Simulation Settings")

model_name = st.sidebar.selectbox("Mathematical Model", ["Boltzmann (Eq 31)", "Gompertz (Eq 32)"], 
                                  help="Defines the symmetric (Boltzmann) or asymmetric (Gompertz) structure of the growth phases.")
func = boltzmann_term if "Boltzmann" in model_name else gompertz_term

n_phases = st.sidebar.number_input("Number of Phases", 1, 10, 2, 
                                   help="Number of sequential growth steps (e.g., 2 for diauxie).")

st.sidebar.markdown("### Global Parameters")
y_i = st.sidebar.number_input("y_i (Start)", value=0.0, 
                              help="Initial value of the response variable at t=0.")
y_f = st.sidebar.number_input("y_f (End)", value=max(1.0, y_i + 0.1), min_value=y_i + 0.001, 
                              help="Final asymptotic value. Must be greater than y_i.")

p_inputs = []; r_true = []; lam_true = []
st.sidebar.markdown("### Parameters per Phase")
last_lam = -0.1 

for j in range(n_phases):
    with st.sidebar.expander(f"Phase {j+1}", expanded=(j == 0)):
        if n_phases == 1:
            p_val = 1.0 
        else:
            p_val = st.number_input(f"Proportion (p{j+1})", min_value=0.01, max_value=1.0, value=1.0/n_phases, 
                                    help="Fraction of total growth for this phase. Sum of all 'p' must be 1.0.", key=f"p_in_{j}")
        
        r = st.number_input(f"Rate (r_max{j+1})", min_value=0.01, value=1.0, key=f"r_{j}",
                            help="Maximum specific reaction rate for this phase.")
        lam_min = last_lam + 0.01
        lam = st.number_input(f"Lag Time (λ{j+1})", min_value=lam_min, value=max(float(j+1), lam_min), key=f"lam_{j}",
                              help="Time delay before this phase begins. Must be strictly increasing.")
        last_lam = lam 
        
        p_inputs.append(p_val)
        r_true.append(r)
        lam_true.append(lam)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
validation_errors = []
total_p = sum(p_inputs)
if n_phases > 1 and abs(total_p - 1.0) > 0.001:
    validation_errors.append(f"❌ Sum of proportions is {total_p:.3f}. It must be exactly 1.0.")
if y_i >= y_f:
    validation_errors.append("❌ y_i must be strictly less than y_f.")

if not validation_errors:
    tmax, t99 = find_saturation_time(y_i, y_f, p_inputs, r_true, lam_true, func)
else:
    tmax = 10.0; t99 = 9.0

# ------------------------------------------------------------
# NOISE & ROUT
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Noise & Execution")
dev_min = st.sidebar.number_input("Noise Min (Abs)", min_value=0.0, value=0.0, 
                                  help="Minimum standard deviation for the added Gaussian noise.")
dev_max = st.sidebar.number_input("Noise Max (Abs)", min_value=0.0, value=0.1, 
                                  help="Maximum standard deviation for the added Gaussian noise.")
outlier_pct = st.sidebar.number_input("Outlier Probability (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5,
                                      help="Percentage of points that will have 3x-6x higher noise than expected.")

n_rep = st.sidebar.number_input("Replicates", 1, 10, 3, 
                                help="Number of experimental repetitions for the same condition.")
n_points = st.sidebar.number_input("Points/Replicate", 10, 500, 50, 
                                   help="Number of data points collected in each replicate.")
n_tests = st.sidebar.number_input("Monte Carlo Tests", 1, 500, 20, 
                                  help="Total number of independent simulations to run.")

st.sidebar.markdown("### Robustness")
outlier_method = st.sidebar.selectbox("Outlier Removal Method", 
                                      ["None", "ROUT-like (Simple MAD)", "ROUT (Robust + FDR)"],
                                      help="Select the strategy to handle outliers before the final fit.")

rout_q = 1.0
if "ROUT (Robust" in outlier_method:
    rout_q = st.sidebar.slider("ROUT Q (Max FDR %)", 0.1, 10.0, 1.0, 
                               help="Maximum desired False Discovery Rate. Lower values make detection stricter.")

# ------------------------------------------------------------
# PREVIEW
# ------------------------------------------------------------
st.subheader("Experimental Design Preview")

if validation_errors:
    for err in validation_errors: st.error(err)
    st.warning("Please fix the errors in the sidebar to proceed.")
    run_btn = st.button("🚀 Run Monte Carlo Simulation", type="primary", disabled=True)
else:
    t_sim = np.linspace(0, tmax, n_points)
    ygen, components_list = polyauxic_components(t_sim, y_i, y_f, p_inputs, r_true, lam_true, func)

    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
        ax_prev.plot(t_sim, ygen, 'k-', lw=2.5, label="Total Curve")
        ax_prev.axvline(t99, color='red', linestyle='--', alpha=0.5, label=f"99% Sat (t={t99:.1f})")
        ax_prev.axvspan(t99, tmax, color='gray', alpha=0.1, label="Plateau (10%)")
        
        if n_phases > 1:
            colors = plt.cm.viridis(np.linspace(0, 1, n_phases))
            for j, (comp_y, color) in enumerate(zip(components_list, colors)):
                ax_prev.plot(t_sim, y_i + comp_y, ls=':', lw=1.5, color=color, label=f"Phase {j+1}")
        
        ax_prev.set_title(f"Generating Function: {model_name}")
        ax_prev.set_xlabel("Time (t)")
        ax_prev.set_ylabel("Response (y)")
        ax_prev.legend(fontsize='small', loc='lower right')
        ax_prev.grid(True, ls=':', alpha=0.6)
        st.pyplot(fig_prev)

    with col_g2:
        st.success("✅ Parameters Validated")
        st.info(f"**Time Calculation:**\n\n"
                f"• Growth Phase (90%): 0 to {t99:.2f}\n"
                f"• Plateau Phase (10%): {t99:.2f} to {tmax:.2f}\n"
                f"• Total Time: {tmax:.2f}")
        df_params = pd.DataFrame({
            "Phase": [f"#{j+1}" for j in range(n_phases)],
            "p": [f"{v:.3f}" for v in p_inputs],
            "r_max": r_true,
            "lambda": lam_true
        })
        st.table(df_params)

    st.markdown("---")
    run_btn = st.button("🚀 Run Monte Carlo Simulation", type="primary")

# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------
if run_btn and not validation_errors:
    df, y_mean, y_std, out_t, out_y, total_points = monte_carlo(
        func, ygen, t_sim, p_inputs, r_true, lam_true,
        dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, 
        outlier_method, outlier_pct, rout_q
    )

    dy = abs(y_f - y_i)
    y_min_plot = min(y_i, y_f) - 0.1*dy
    y_max_plot = max(y_i, y_f) + 0.1*dy

    st.markdown("## Simulation Results")
    
    # Graphs and Tables
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_sim, ygen, 'k-', lw=2, label="True Curve", zorder=3)
    ax.errorbar(t_sim, y_mean, yerr=y_std, fmt='o', color='royalblue', 
                ecolor='lightblue', alpha=0.6, capsize=0, markersize=3, label="Sim Mean ± Std", zorder=2)
    
    # PLOT DETECTED OUTLIERS
    if len(out_t) > 0:
        ax.scatter(out_t, out_y, marker='x', color='red', s=40, label=f"Detected Outliers", zorder=4)
        
    ax.set_ylim(min(y_min_plot, -0.05), y_max_plot)
    ax.set_xlabel("Time"); ax.set_ylabel("y")
    ax.grid(True, ls=':'); ax.legend()
    st.markdown("### Uncertainty Envelope & Outliers")
    st.pyplot(fig)
    
    # Metrics below graph
    m1, m2 = st.columns(2)
    m1.metric("Total Simulated Points", total_points)
    m2.metric("Total Outliers Removed", len(out_t), delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Fit Quality (Adj R²)")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["test"], df["R2_adj"], color='green', marker='s', ms=4, label="Adj R²")
        # Inserido R2 simples aqui também como solicitado no texto (embora o label diga Adj R2, vou plotar ambos)
        ax.plot(df["test"], df["R2"], color='lightgreen', marker='.', ms=3, label="R²", alpha=0.7)
        ax.set_xlabel("Test Index"); ax.set_ylabel("Adjusted R²")
        ax.legend()
        ax.grid(True, ls=':')
        st.pyplot(fig)
    with c2:
        st.markdown("#### Information Criteria")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["test"], df["AIC"], marker='o', ms=4, label="AIC")
        ax.plot(df["test"], df["AICc"], marker='D', ms=3, label="AICc") # Adicionado AICc
        ax.plot(df["test"], df["BIC"], marker='^', ms=4, label="BIC")
        ax.set_xlabel("Test Index"); ax.set_ylabel("Value")
        ax.legend(); ax.grid(True, ls=':')
        st.pyplot(fig)

    st.markdown("#### Data Table")
    st.dataframe(df.head(10))
    st.download_button("Download Results (CSV)", df.to_csv(index=False), "monte_carlo_results.csv", "text/csv")

    st.markdown("---")
    st.markdown("### 1. Statistical Parameter Distribution (Boxplots)")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].boxplot([df["yi_hat"], df["yf_hat"]], labels=["yi", "yf"])
    axs[0,0].set_title("Boundary Parameters")
    axs[0,1].boxplot([df[f"p{j+1}"] for j in range(n_phases)], labels=[f"p{j+1}" for j in range(n_phases)])
    axs[0,1].set_title("Proportions (p)")
    axs[1,0].boxplot([df[f"r{j+1}"] for j in range(n_phases)], labels=[f"r{j+1}" for j in range(n_phases)])
    axs[1,0].set_title("Rates (r_max)")
    axs[1,1].boxplot([df[f"lam{j+1}"] for j in range(n_phases)], labels=[f"λ{j+1}" for j in range(n_phases)])
    axs[1,1].set_title("Lags (λ)")
    for ax in axs.flat: ax.grid(True, ls=':', alpha=0.5)
    st.pyplot(fig)

    st.markdown("### 2. Parameter Behavior per Test (Stability)")
    st.caption("Solid lines: Fitted values | Dashed lines: True input values used in generation")
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    
    axs2[0,0].plot(df["test"], df["yi_hat"], label="Fitted yi", marker='.', color='C0')
    axs2[0,0].axhline(y_i, color='C0', linestyle='--', alpha=0.7, label="True yi")
    axs2[0,0].plot(df["test"], df["yf_hat"], label="Fitted yf", marker='.', color='C1')
    axs2[0,0].axhline(y_f, color='C1', linestyle='--', alpha=0.7, label="True yf")
    axs2[0,0].set_title("Start (yi) & End (yf)"); axs2[0,0].legend()
    
    for j in range(n_phases):
        line, = axs2[0,1].plot(df["test"], df[f"p{j+1}"], label=f"p{j+1}", marker='.')
        axs2[0,1].axhline(p_inputs[j], color=line.get_color(), linestyle='--', alpha=0.7)
    axs2[0,1].set_title("Proportions (p)"); axs2[0,1].legend()

    for j in range(n_phases):
        line, = axs2[1,0].plot(df["test"], df[f"r{j+1}"], label=f"r{j+1}", marker='.')
        axs2[1,0].axhline(r_true[j], color=line.get_color(), linestyle='--', alpha=0.7)
    axs2[1,0].set_title("Rates (r_max)"); axs2[1,0].legend()

    for j in range(n_phases):
        line, = axs2[1,1].plot(df["test"], df[f"lam{j+1}"], label=f"λ{j+1}", marker='.')
        axs2[1,1].axhline(lam_true[j], color=line.get_color(), linestyle='--', alpha=0.7)
    axs2[1,1].set_title("Lags (λ)"); axs2[1,1].legend()

    for ax in axs2.flat: 
        ax.grid(True, ls=':', alpha=0.5); ax.set_xlabel("Test Index")
    plt.tight_layout()
    st.pyplot(fig2)
import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components
# ==============================================================================
# REFERENCE SECTION (Paper, Altmetric, Project GitHub) & FOOTER
# ==============================================================================
TEXTS = {
    'paper_ref': 'Paper Reference'
}
profile_pic_url = "https://github.com/gusmock.png"

st.markdown("---")
st.subheader(f"📄 {TEXTS['paper_ref']}")

# HTML for the reference card
# Changes: Font is now sans-serif, and container is optimized for spacing
ref_html = """
<style>
    .ref-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        gap: 15px;
        /* Standard sans-serif font */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        flex-wrap: wrap;
    }
    .citation-text {
        font-size: 16px;
        color: #333;
        font-weight: 500;
        margin-right: auto;
        line-height: 1.4;
    }
    .badge-group {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .badge-group img {
        height: 22px;
        transition: transform 0.2s;
    }
    .badge-group img:hover {
        transform: scale(1.05);
    }
    a { text-decoration: none; }
</style>

<div class="ref-container">
    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-arxiv-id='2507.05960' data-hide-no-mentions='true'></div>
    
    <div class="citation-text">
        Mockaitis, G. (2025) <strong>Mono and Polyauxic Growth Kinetic Models</strong>. <br>
        <span style="color: #666; font-size: 14px;">ArXiv: 2507.05960, 24 p.</span>
    </div>

    <div class="badge-group">
        <a href="https://doi.org/10.48550/arXiv.2507.05960" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2507.05960-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
        </a>
        <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Repo">
        </a>
    </div>

    <script type='text/javascript' src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>
</div>
"""

# Increased height to 150 to prevent the bottom from being cut off
components.html(ref_html, height=150)

st.markdown("---")

footer_html = f"""
<style>
    /* Main Footer Container */
    .footer-container {{
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #444;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px 0;
    }}
    
    /* Photo and Text Area */
    .profile-section {{
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
        max-width: 800px;
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 600px) {{
        .profile-section {{
            flex-direction: column;
        }}
    }}

    .profile-img {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .profile-info {{
        text-align: left;
    }}
    
    @media (max-width: 600px) {{
        .profile-info {{ text-align: center; }}
    }}

    .profile-info h2 {{
        margin: 0;
        font-size: 16px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .profile-info h4 {{
        margin: 5px 0;
        font-size: 18px;
        color: #222;
        font-weight: 700;
    }}
    
    .profile-info p {{
        margin: 0;
        font-size: 13px;
        color: #666;
        line-height: 1.4;
    }}

    /* Personal Badges Container */
    .social-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
    }}
    
    .social-badges a img {{
        height: 26px;
        border-radius: 4px;
        transition: transform 0.2s, opacity 0.2s;
    }}
    
    .social-badges a img:hover {{
        transform: translateY(-2px);
        opacity: 0.9;
    }}
</style>

<div class="footer-container">
    
    <div class="profile-section">
        <img src="{profile_pic_url}" class="profile-img" alt="Gustavo Mockaitis">
        
        <div class="profile-info">
            <h2>GBMA / FEAGRI / UNICAMP</h2>
            <h4>Dev: Prof. Dr. Gustavo Mockaitis</h4>
            <p>
                Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and Environment<br>
                School of Agricultural Engineering, University of Campinas.<br>
                Campinas, SP, Brazil.
            </p>
        </div>
    </div>

    <div class="social-badges">
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank">
            <img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID">
        </a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=en&oi=ao" target="_blank">
            <img src="https://img.shields.io/badge/Scholar-Profile-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar">
        </a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/ResearchGate-Profile-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate">
        </a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank">
            <img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes CV">
        </a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank">
            <img src="https://img.shields.io/badge/Web_of_Science-Profile-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science">
        </a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/UNICAMP-Institutional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP">
        </a>
    </div>

</div>
"""

components.html(footer_html, height=280, scrolling=False)
