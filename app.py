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
POLYAUXIC ROBUSTNESS SIMULATOR (v3.0 - ROUT Enabled)
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)
GitHub: https://github.com/gusmock/mono_polyauxic_kinetics/

DESCRIPTION:
This Streamlit application performs rigorous Monte Carlo robustness testing for 
Polyauxic Kinetic Models. It incorporates the ROUT method (Robust regression 
and Outlier removal) based on Motulsky & Brown (2006) to handle outliers 
scientifically using False Discovery Rate (FDR).

REFERENCES:
1. Motulsky, H. J., & Brown, R. E. (2006). Detecting outliers when fitting 
   data with nonlinear regression – a new method based on robust nonlinear 
   regression and the false discovery rate. BMC Bioinformatics, 7, 123.
   https://doi.org/10.1186/1471-2105-7-123

2. Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute 
   Deviation. Journal of the American Statistical Association.

================================================================================
PIPELINE WITH ROUT
================================================================================

1. DATA GENERATION:
   - True Curve + Heteroscedastic Noise + Stochastic Outliers (contaminants).

2. ROBUST FIT (Step A of ROUT):
   - The model is fitted using a Robust Loss function (Soft L1/Lorentzian).
   - This ensures the curve ignores outliers and follows the main trend.

3. OUTLIER DETECTION (Step B of ROUT):
   - Absolute residuals are calculated from the Robust Fit.
   - The Robust Standard Deviation of Residuals (RSDR) is estimated via MAD.
   - A critical threshold is calculated based on the user-defined Q (FDR) 
     and the sample size N, using the t-distribution.
   - Points exceeding this threshold are flagged.

4. FINAL FIT (Step C):
   - The flagged points are removed.
   - A standard Least Squares (SSE) fit is performed on the "clean" data 
     to obtain unbiased parameters and valid confidence intervals.
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
    Fast but susceptible to masking effects from large outliers.
    """
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5

def detect_outliers_rout_rigorous(y_true, y_pred, Q=1.0):
    """
    ROUT (Rigorous): Uses residuals from Robust Fit.
    Threshold: Calculated dynamically based on FDR (Q) and t-distribution.
    Resistant to masking effects.
    """
    residuals = y_true - y_pred
    abs_res = np.abs(residuals)
    
    # Robust Standard Deviation (RSDR)
    med_res = np.median(abs_res)
    rsdr = 1.4826 * med_res if med_res > 1e-12 else 1e-12
    
    # Normalize residuals
    t_scores = abs_res / rsdr
    
    # Critical Threshold (Approximate correction for FDR)
    n = len(y_true)
    # Adjustment for False Discovery Rate: alpha is derived from Q
    alpha = (Q / 100.0) 
    
    # Critical t value
    critical_t = t_dist.ppf(1 - (alpha / 2), df=n-1)
    
    return t_scores > critical_t

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

def sse_loss(theta, t, y, func, n_phases):
    if theta[0] >= theta[1]: return 1e12 
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

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
    # Storage for detected outliers (t, y) to plot later
    detected_outliers_t = []
    detected_outliers_y = []
    
    if outlier_method == "None":
        th, met = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        
    elif outlier_method == "ROUT-like (Simple MAD)":
        # 1. Fit (SSE) -> 2. Detect (MAD Z>2.5) -> 3. Refit (SSE)
        th_pre, _ = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        y_pred_pre = polyauxic_fit_model(t_all, th_pre, func, n_phases)
        
        mask = detect_outliers_mad_simple(y_all, y_pred_pre) # Returns True for outliers
        
        # Save outliers for plotting
        if np.any(mask):
            detected_outliers_t = t_all[mask]
            detected_outliers_y = y_all[mask]
            
        t_clean = t_all[~mask]; y_clean = y_all[~mask]
        
        if len(y_clean) < len(th_pre) + 5: 
             th, met = fit_polyauxic(t_all, y_all, func, n_phases, loss_function=sse_loss)
        else:
             th, met = fit_polyauxic(t_clean, y_clean, func, n_phases, loss_function=sse_loss)
             
    elif outlier_method == "ROUT (Robust + FDR)":
        # 1. Fit (Robust) -> 2. Detect (ROUT Q) -> 3. Refit (SSE)
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
    # To save memory, we calculate mean/std incrementally or just stack everything if N is manageable.
    # For N_tests=100 and points=50, stacking is fine.
    all_t_flat = []
    all_y_flat = []
    all_out_t = []
    all_out_y = []
    
    # We also need a matrix for the "Mean +/- Std" calculation.
    # Logic: Since time points t_sim are fixed for generation, we can stack raw matrices from each test.
    # Each test produces (n_rep, n_points). We stack them to get (n_tests * n_rep, n_points).
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
            
            # Aggregate data for plotting
            all_t_flat.append(t_dat)
            all_y_flat.append(y_dat)
            
            # For Mean/Std calculation (we only need the values corresponding to t_sim steps)
            # t_dat is [t_sim, t_sim, t_sim], y_dat is [y_rep1, y_rep2, y_rep3]
            # Reshape y_dat to (n_rep, n_points)
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
    
    # Process aggregates for plotting
    final_y_matrix = np.vstack(y_matrix_stack) # Shape: (Total_Reps, N_Points)
    y_mean = np.mean(final_y_matrix, axis=0)
    y_std = np.std(final_y_matrix, axis=0)
    
    # Flatten outlier lists
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
        ax.plot(df["test"], df["R2_adj"], color='green', marker='s', ms=4)
        ax.set_xlabel("Test Index"); ax.set_ylabel("Adjusted R²")
        ax.grid(True, ls=':')
        st.pyplot(fig)
    with c2:
        st.markdown("#### Information Criteria")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["test"], df["AIC"], marker='o', ms=4, label="AIC")
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

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")

footer_html = """
<style>
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
    }
    .footer-text {
        width: 100%;
        text-align: center;
        font-family: sans-serif;
        color: #444;
        margin-bottom: 20px;
    }
    img {
        height: 28px;
    }
</style>

<div class="footer-text">
    <h4 style="margin-bottom: 5px; margin-top: 0;">Desenvolvido por: Prof. Dr. Gustavo Mockaitis</h4>
    <p style="margin-top: 0; font-size: 0.9em;">GBMA / FEAGRi / UNICAMP</p>

    <div class="badge-container">
        <a href="https://arxiv.org/abs/2507.05960" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2507.05960-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
        </a>
        <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank">
            <img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID">
        </a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=pt-BR&oi=ao" target="_blank">
            <img src="https://img.shields.io/badge/Scholar-Perfil-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar">
        </a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/ResearchGate-Perfil-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate">
        </a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank">
            <img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes">
        </a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Conectar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank">
            <img src="https://img.shields.io/badge/Web_of_Science-Perfil-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science">
        </a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/UNICAMP-Institucional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP">
        </a>
    </div>
</div>
"""

st_components.html(footer_html, height=350, scrolling=False)
