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
Polyauxic Robustness Simulator
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)

This Streamlit application performs Monte Carlo robustness testing
for polyauxic Boltzmann (Eq. 31) and Gompertz (Eq. 32) models.

The procedure follows exactly this sequence:

1) The user selects:
   - Model (Boltzmann or Gompertz)
   - Number of phases
   - True parameters for simulation (y_i, y_f; p_j, r_max_j, lambda_j)
   - Noise range (absolute deviation min/max)
   - Number of replicates
   - Points per replicate
   - Number of Monte Carlo tests
   - Whether to apply ROUT-like outlier removal

2) A **generating function** (noise-free curve) is computed once:
       y_gen(t) = y_i + (y_f - y_i) * sum_j term_j(t)

   Before this step, the following constraints are enforced:
       - y_i < y_f
       - p_j > 0 for all j and Σ p_j = 1 (they are renormalized)
       - lambda_1 < lambda_2 < ... < lambda_n

3) For each Monte Carlo test:
   - Each replicate receives a new independent noise realization:
         noise = scale * Normal(0,1)
         where scale = dev_min + (dev_max - dev_min)*Uniform(0,1)
   - All replicates are concatenated for fitting.

4) Fitting uses the same method as your main kinetic platform:
   Differential Evolution → L-BFGS-B,
   with softmax parametrization for p_j.

5) Metrics recorded for each test:
   - Fitted parameters (y_i, y_f, p_j, r_max_j, lambda_j)
   - SSE, R², Adjusted R²
   - AIC, AICc, BIC

6) Output:
   - Graph showing the generating function (noise-free)
   - Graph showing the generating function + global mean ± std of all
     simulated datasets (all tests × all replicates)
   - Table of all Monte Carlo results
   - Criteria plots (AIC, AICc, BIC vs test)
   - R² plots (R², R²_adj vs test)
   - Four parameter plots arranged as 2×2:
         (1) yi,yf    (2) p_j    (3) r_max_j    (4) lambda_j
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Polyauxic Robustness Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 0. ROUT-like Outlier Detection (MAD-based)
# ------------------------------------------------------------
def detect_outliers(y_true, y_pred):
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5

# ------------------------------------------------------------
# 1. Model Equations
# ------------------------------------------------------------
def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    # Avoid overflow/division by zero
    expo = (4 * r_j * (lam_j - t)) / (dy * p) + 2
    expo = np.clip(expo, -500, 500) 
    return p / (1 + np.exp(expo))

def gompertz_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    expo = (r_j * np.e * (lam_j - t)) / (dy * p) + 1
    expo = np.clip(expo, -500, 500)
    return p * np.exp(-np.exp(expo))

# ------------------------------------------------------------
# 2. Polyauxic generating function
# ------------------------------------------------------------
def polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    p_vec = np.asarray(p_vec, dtype=float)
    # Normalize proportions if invalid
    if np.sum(p_vec) <= 0:
        p_vec = np.ones_like(p_vec) / len(p_vec)
    else:
        p_vec = p_vec / np.sum(p_vec)
    
    components = []
    sum_terms = np.zeros_like(t, dtype=float)
    
    for j in range(len(p_vec)):
        term = func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
        components.append((y_f - y_i) * term)
        sum_terms += term
        
    y_total = y_i + (y_f - y_i) * sum_terms
    return y_total, components, p_vec

def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    y, _, _ = polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func)
    return y

# ------------------------------------------------------------
# 3. Fitting & Helpers
# ------------------------------------------------------------
def polyauxic_fit_model(t, theta, func, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r = theta[2 + n_phases : 2 + 2*n_phases]
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]
    
    # Softmax for p
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    p = expz / np.sum(expz)
    
    sum_terms = np.zeros_like(t)
    for j in range(n_phases):
        sum_terms += func(t, y_i, y_f, p[j], r[j], lam[j])
    return y_i + (y_f - y_i) * sum_terms

def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    # Penalty for extreme negative predictions
    if np.any(y_pred < -0.1*np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

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
    theta0[0] = max(0, y.min()) # Ensure non-negative start guess
    theta0[1] = y.max()
    for i,(lam,r) in enumerate(guesses):
        theta0[2 + n_phases + i] = r
        theta0[2 + 2*n_phases + i] = lam
    return theta0

def fit_polyauxic(t_all, y_all, func, n_phases):
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1
    t_n = t_all / t_scale
    y_n = y_all / y_scale
    
    theta0 = smart_guess(t_all, y_all, n_phases)
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale; th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0 
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases]*(t_scale/y_scale)
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases]/t_scale
    
    # ---------------------------------------------------------
    # CONSTRAINT: yi >= 0
    # The lower bound for the first parameter (yi normalized) is set to 0.0
    # ---------------------------------------------------------
    bounds = [(0.0, 1.5), (0, 2)] + [(-10,10)]*n_phases + [(0,500)]*n_phases + [(-0.1,1.2)]*n_phases
    
    popsize=20
    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))
    
    res_de = differential_evolution(sse_loss, bounds, args=(t_n,y_n,func,n_phases),
                                    maxiter=800, popsize=popsize, init=init_pop, strategy="best1bin", polish=True, tol=1e-6)
    res = minimize(sse_loss, res_de.x, args=(t_n,y_n,func,n_phases), method="L-BFGS-B", bounds=bounds, tol=1e-10)
    
    th_n = res.x
    th = np.zeros_like(th_n)
    th[0]=th_n[0]*y_scale; th[1]=th_n[1]*y_scale
    th[2:2+n_phases]=th_n[2:2+n_phases]
    th[2+n_phases:2+2*n_phases]=th_n[2+n_phases:2+2*n_phases]*(y_scale/t_scale)
    th[2+2*n_phases:2+3*n_phases]=th_n[2+2*n_phases:2+3*n_phases]*t_scale
    
    y_pred = polyauxic_fit_model(t_all, th, func, n_phases)
    sse = np.sum((y_all - y_pred)**2)
    sst = np.sum((y_all - np.mean(y_all))**2)
    r2 = 1 - sse/sst if sst>0 else np.nan
    n = len(y_all); k = len(th)
    r2adj = 1 - (1-r2)*(n-1)/(n-k-1) if (n-k-1)>0 else np.nan
    aic = n*np.log(sse/n) + 2*k
    bic = n*np.log(sse/n) + k*np.log(n)
    aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf
    return th, {"SSE":sse,"R2":r2,"R2_adj":r2adj,"AIC":aic,"AICc":aicc,"BIC":bic}

# ------------------------------------------------------------
# 4. Monte Carlo Engine
# ------------------------------------------------------------
def monte_carlo_single(test_idx, func, ygen, t_sim, p_true, r_true, lam_true,
                       dev_min, dev_max, n_rep, n_points, y_i, y_f, use_rout):
    n_phases = len(p_true)
    t_all_list = []; y_all_list = []
    y_matrix = np.zeros((n_rep, n_points))
    for rep in range(n_rep):
        scales = dev_min + (dev_max-dev_min)*np.random.rand(n_points)
        noise = scales * np.random.normal(0,1,n_points)
        y_obs = ygen + noise
        t_all_list.append(t_sim); y_all_list.append(y_obs)
        y_matrix[rep, :] = y_obs
    t_all = np.concatenate(t_all_list); y_all = np.concatenate(y_all_list)
    
    if use_rout:
        th_pre, _ = fit_polyauxic(t_all, y_all, func, n_phases)
        y_pred_pre = polyauxic_fit_model(t_all, th_pre, func, n_phases)
        mask = detect_outliers(y_all, y_pred_pre)
        t_clean = t_all[~mask]; y_clean = y_all[~mask]
        if len(y_clean) < len(th_pre) + 2: 
             th, met = fit_polyauxic(t_all, y_all, func, n_phases)
        else:
             th, met = fit_polyauxic(t_clean, y_clean, func, n_phases)
    else:
        th, met = fit_polyauxic(t_all, y_all, func, n_phases)
        
    yi_hat = th[0]; yf_hat = th[1]
    z = th[2:2+n_phases]; r = th[2+n_phases:2+2*n_phases]; lam = th[2+2*n_phases:2+3*n_phases]
    z_shift = z-np.max(z)
    p_hat = np.exp(z_shift)/np.sum(np.exp(z_shift))
    
    row = {"test":test_idx, "yi_hat":yi_hat, "yf_hat":yf_hat,
           "SSE":met["SSE"],"R2":met["R2"],"R2_adj":met["R2_adj"],
           "AIC":met["AIC"],"AICc":met["AICc"],"BIC":met["BIC"]}
    for j in range(n_phases):
        row[f"p{j+1}"]=p_hat[j]; row[f"r{j+1}"]=r[j]; row[f"lam{j+1}"]=lam[j]
    return row, y_matrix

def monte_carlo(func, ygen, t_sim, p_true, r_true, lam_true,
                dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, use_rout):
    results = []; all_y_blocks = []
    progress = st.progress(0.0)
    status_text = st.empty()
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(monte_carlo_single, i+1, func, ygen, t_sim, p_true, r_true, lam_true,
                                   dev_min, dev_max, n_rep, n_points, y_i, y_f, use_rout): i+1 for i in range(n_tests)}
        done = 0
        for fut in as_completed(futures):
            row, y_mat = fut.result()
            results.append(row); all_y_blocks.append(y_mat)
            done += 1
            progress.progress(done / n_tests)
            status_text.text(f"Simulating Test {done}/{n_tests}...")
    status_text.text("Simulation finished successfully.")
    df = pd.DataFrame(results).sort_values("test")
    all_y = np.vstack(all_y_blocks)
    return df, np.mean(all_y, axis=0), np.std(all_y, axis=0)

# ------------------------------------------------------------
# 5. STREAMLIT APP - INTERFACE & LAYOUT
# ------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>Polyauxic Robustness Simulator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Robustness analysis tool for polyauxic kinetic models via Monte Carlo</p>", unsafe_allow_html=True)

# --- INSTRUCTIONS GUIDE (Dropdown) ---
with st.expander("📘 Parameter Guide & Instructions (Click to open)"):
    st.markdown("""
    This simulator allows testing the capability of **Boltzmann** and **Gompertz** models to recover kinetic parameters from noisy data.
    
    ### ⚙️ Model Parameters
    * **Mathematical Model:** Choose between *Boltzmann* (symmetric sigmoidal) or *Gompertz* (asymmetric sigmoidal).
    * **Number of Phases:** How many growth phases occur sequentially (e.g., diauxie = 2 phases).
    * **y_i (Start):** Initial value of the response variable (e.g., initial biomass). **Constraint:** The fitted model forces $y_i \ge 0$.
    * **y_f (End):** Maximum final value reached. Must be greater than $y_i$.
    
    ### 📊 Phase Parameters
    * **Raw Proportion (p):** Relative contribution of the phase. The sum will be automatically normalized to 100%.
    * **Rate (r_max):** Maximum specific reaction rate for that phase.
    * **Lag Time (λ):** Delay time for the start of the phase. Must be sequential ($\lambda_1 < \lambda_2 < ...$).
    
    ### 🎲 Noise & Monte Carlo Settings
    * **Noise Min/Max (Abs):** Defines the range of the noise standard deviation. Each replicate will have a random noise level within this range.
    * **Replicates:** Number of experimental repetitions (triplicates, etc.) per test.
    * **Points/Rep:** Number of time points collected in each replicate.
    * **MC Tests:** Total number of independent "virtual experiments" to simulate.
    * **Remove Outliers (ROUT):** If enabled, uses a MAD-based algorithm to remove statistical outliers before the final fit.
    """)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Simulation Settings")

model = st.sidebar.selectbox("Mathematical Model", ["Boltzmann (Eq 31)", "Gompertz (Eq 32)"])
func = boltzmann_term if "Boltzmann" in model else gompertz_term

n_phases = st.sidebar.number_input("Number of Phases", 1, 10, 2)

st.sidebar.markdown("### Global Parameters")
y_i = st.sidebar.number_input("y_i (Start)", value=0.0)
y_f_min = y_i + 0.01
y_f = st.sidebar.number_input("y_f (End)", min_value=y_f_min, value=max(1.0, y_f_min))

p_inputs = []; r_true = []; lam_true = []
st.sidebar.markdown("### Parameters per Phase")
last_lam = -0.1 

for j in range(n_phases):
    with st.sidebar.expander(f"Phase {j+1}", expanded=(j == 0)):
        p_in = st.number_input(f"Raw Proportion (p{j+1})", min_value=0.01, value=1.0, key=f"p_in_{j}")
        r = st.number_input(f"Rate (r_max{j+1})", min_value=0.01, value=1.0, key=f"r_{j}")
        lam_min = last_lam + 0.1
        lam = st.number_input(f"Lag Time (λ{j+1})", min_value=lam_min, value=max(float(j+1), lam_min), key=f"lam_{j}")
        last_lam = lam 
        p_inputs.append(p_in)
        r_true.append(r)
        lam_true.append(lam)

total_p = sum(p_inputs)
p_true = [p / total_p for p in p_inputs]

st.sidebar.markdown("---")
st.sidebar.markdown("### Noise & Execution")
dev_min = st.sidebar.number_input("Noise Min (Abs)", min_value=0.0, value=0.0)
dev_max = st.sidebar.number_input("Noise Max (Abs)", min_value=0.0, value=0.1)
n_rep = st.sidebar.number_input("Replicates", 1, 10, 3)
n_points = st.sidebar.number_input("Points/Replicate", 10, 500, 50)
n_tests = st.sidebar.number_input("Monte Carlo Tests", 1, 500, 20)
use_rout = st.sidebar.checkbox("Remove Outliers (ROUT)", value=False)

# ------------------------------------------------------------
# LIVE PREVIEW
# ------------------------------------------------------------
st.subheader("True Parameters Preview")

max_lam = max(lam_true)
tmax = max(3*max_lam, 1.0)
t_sim = np.linspace(0, tmax, n_points)
ygen, components, p_norm_vec = polyauxic_components(t_sim, y_i, y_f, p_true, r_true, lam_true, func)

col_g1, col_g2 = st.columns([2, 1])

with col_g1:
    fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
    ax_prev.plot(t_sim, ygen, 'k-', lw=2.5, label="Total Curve")
    colors = plt.cm.viridis(np.linspace(0, 1, n_phases))
    for j, (comp_y, color) in enumerate(zip(components, colors)):
        ax_prev.plot(t_sim, y_i + comp_y, ls='--', lw=1.5, color=color, label=f"Phase {j+1}")
    ax_prev.set_title(f"Generating Function ({model})")
    ax_prev.set_xlabel("Time (t)")
    ax_prev.set_ylabel("Response (y)")
    ax_prev.grid(True, ls=':', alpha=0.6)
    ax_prev.legend(fontsize='small')
    st.pyplot(fig_prev)

with col_g2:
    st.markdown("**Effective Parameters:**")
    st.info(f"Optimization set with constraint: **yi_hat >= 0**")
    df_params = pd.DataFrame({
        "Phase": [f"#{j+1}" for j in range(n_phases)],
        "p (norm)": [f"{v:.3f}" for v in p_norm_vec],
        "r_max": r_true,
        "lambda": lam_true
    })
    st.table(df_params)

# ------------------------------------------------------------
# MONTE CARLO EXECUTION
# ------------------------------------------------------------
st.markdown("---")
if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
    
    df, y_mean, y_std = monte_carlo(
        func, ygen, t_sim, p_true, r_true, lam_true,
        dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, use_rout
    )

    dy = abs(y_f - y_i) if abs(y_f - y_i) > 0 else 1.0
    y_min_plot = min(y_i, y_f) - 0.1*dy
    y_max_plot = max(y_i, y_f) + 0.1*dy

    st.markdown("## Simulation Results")
    
    # 1. Aggregate Graph
    st.markdown("### Uncertainty Envelope (Global Mean ± Std)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_sim, ygen, 'k--', lw=2, label="True Curve")
    ax.errorbar(t_sim, y_mean, yerr=y_std, fmt='o', color='royalblue', 
                ecolor='lightblue', alpha=0.5, capsize=0, markersize=3, label="Simulated Mean ± Std")
    ax.set_ylim(min(y_min_plot, -0.05), y_max_plot)
    ax.set_xlabel("Time")
    ax.set_ylabel("y")
    ax.grid(True, ls=':')
    ax.legend()
    st.pyplot(fig)

    # 2. Tables and Metrics
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

    # 3. Parameter Distribution (Boxplots)
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

    # 4. Behavior per Test (Line Plots)
    st.markdown("### 2. Parameter Behavior per Test (Stability)")
    st.caption("X-Axis = Monte Carlo Test Index | Y-Axis = Fitted Value")
    
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
    
    # Yi and Yf
    axs2[0,0].plot(df["test"], df["yi_hat"], label="yi_hat", marker='.')
    axs2[0,0].plot(df["test"], df["yf_hat"], label="yf_hat", marker='.')
    axs2[0,0].set_title("yi and yf")
    axs2[0,0].legend()
    
    # Proportions
    for j in range(n_phases):
        axs2[0,1].plot(df["test"], df[f"p{j+1}"], label=f"p{j+1}", marker='.')
    axs2[0,1].set_title("Proportions (p)")
    axs2[0,1].legend()

    # Rates
    for j in range(n_phases):
        axs2[1,0].plot(df["test"], df[f"r{j+1}"], label=f"r{j+1}", marker='.')
    axs2[1,0].set_title("Rates (r_max)")
    axs2[1,0].legend()

    # Lags
    for j in range(n_phases):
        axs2[1,1].plot(df["test"], df[f"lam{j+1}"], label=f"λ{j+1}", marker='.')
    axs2[1,1].set_title("Lags (λ)")
    axs2[1,1].legend()

    for ax in axs2.flat: 
        ax.grid(True, ls=':', alpha=0.5)
        ax.set_xlabel("Test Index")
    
    st.pyplot(fig2)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <h4>Developed by: Prof. Dr. Gustavo Mockaitis</h4>
        <p>GBMA / FEAGRi / UNICAMP</p>
        <p>
            <a href='https://github.com/gusmock/mono_polyauxic_kinetics/tree/Simulator' target='_blank'>GitHub Repository</a> | 
            ORCID | ResearcherID | Lattes
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
