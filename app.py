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

Polyauxic Robustness Simulator (FINAL VERSION)
==============================================

Implements Monte Carlo robustness assessment for Boltzmann (Eq.31)
and Gompertz (Eq.32) polyauxic kinetic models.

FEATURES:
---------
1) User chooses:
   - Model (Boltzmann or Gompertz)
   - Number of phases
   - True simulation parameters (y_i, y_f, p_j, r_j, lambda_j)
   - Absolute deviation min/max
   - Number of replicates
   - Points per replicate
   - Number of Monte Carlo tests
   - Whether to apply ROUT-like outlier detection

2) A *generating function* y_gen(t) is created (noise-free reference).

3) For each Monte Carlo test:
   - Noise is added:  
         noise = scale * Normal(0,1)
     where scale ~ Uniform(dev_min, dev_max)

   - Pre-fit → Outlier detection (if enabled) → Remove → Final fit

4) Outputs:
   - Plot 1: Generating function
   - Plot 2: Global mean ± std of all simulated points
   - Monte Carlo results table + CSV download
   - AIC / AICc / BIC plots
   - R² / R²_adj plots
   - Parameter behaviors in a 2×2 layout:
       (yi,yf), (p_j), (r_j), (lambda_j)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks


# ------------------------------------------------------------
# 0. ROUT OUTLIER DETECTION (same logic as original app)
# ------------------------------------------------------------

def detect_outliers(y_true, y_pred):
    """ROUT-like MAD detection as in main polyauxic app."""
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5


# ------------------------------------------------------------
# 1. Model equations
# ------------------------------------------------------------

def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
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

def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    p_vec = np.asarray(p_vec)
    if np.sum(p_vec) <= 0:
        p_vec = np.ones_like(p_vec) / len(p_vec)
    else:
        p_vec = p_vec / np.sum(p_vec)

    out = np.zeros_like(t)
    for j in range(len(p_vec)):
        out += func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
    return y_i + (y_f - y_i) * out


# ------------------------------------------------------------
# 3. Polyauxic model for fitting with softmax
# ------------------------------------------------------------

def polyauxic_fit_model(t, theta, func, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2+n_phases]
    r = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))

    out = np.zeros_like(t)
    for j in range(n_phases):
        out += func(t, y_i, y_f, p[j], r[j], lam[j])

    return y_i + (y_f - y_i) * out


def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1*np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred)**2)


# ------------------------------------------------------------
# 4. Smart initial guess
# ------------------------------------------------------------

def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    if len(dy) >= 5:
        dy_s = np.convolve(dy, np.ones(5)/5, mode="same")
    else:
        dy_s = dy.copy()

    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guesses=[]
    if len(peaks)>0:
        idx = np.argsort(props["peak_heights"])[::-1][:n_phases]
        for p in peaks[idx]:
            guesses.append((t[p], abs(dy_s[p])))

    while len(guesses)<n_phases:
        span = t.max()-t.min() if t.max()>t.min() else 1
        guesses.append((t.min()+span*(len(guesses)+1)/(n_phases+1),
                        (y.max()-y.min())/(span/n_phases)))

    guesses = sorted(guesses, key=lambda x: x[0])

    th = np.zeros(2+3*n_phases)
    th[0] = y.min()
    th[1] = y.max()
    for i,(lam,r) in enumerate(guesses):
        th[2+n_phases+i]=r
        th[2+2*n_phases+i]=lam
    return th


# ------------------------------------------------------------
# 5. Fit engine DE → L-BFGS-B
# ------------------------------------------------------------

def fit_polyauxic(t_all, y_all, func, n_phases):
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1

    t_n = t_all / t_scale
    y_n = y_all / y_scale

    theta0 = smart_guess(t_all, y_all, n_phases)

    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale
    th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases] * t_scale / y_scale
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases] / t_scale

    bounds=[(-0.2,1.5),(0,2)]
    bounds += [(-10,10)]*n_phases
    bounds += [(0,500)]*n_phases
    bounds += [(-0.1,1.2)]*n_phases

    popsize=20
    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))

    res_de = differential_evolution(
        sse_loss, bounds,
        args=(t_n,y_n,func,n_phases),
        maxiter=600, popsize=popsize, strategy="best1bin",
        init=init_pop, polish=True, tol=1e-6
    )
    res = minimize(
        sse_loss, res_de.x,
        args=(t_n,y_n,func,n_phases),
        method="L-BFGS-B", bounds=bounds, tol=1e-10
    )

    thn = res.x
    th = np.zeros_like(thn)
    th[0] = thn[0]*y_scale
    th[1] = thn[1]*y_scale
    th[2:2+n_phases] = thn[2:2+n_phases]
    th[2+n_phases:2+2*n_phases] = thn[2+n_phases:2+2*n_phases]*(y_scale/t_scale)
    th[2+2*n_phases:2+3*n_phases] = thn[2+2*n_phases:2+3*n_phases]*t_scale

    y_pred = polyauxic_fit_model(t_all, th, func, n_phases)
    sse = np.sum((y_all-y_pred)**2)
    sst = np.sum((y_all-np.mean(y_all))**2)
    r2 = 1 - sse/sst if sst>0 else np.nan

    n = len(y_all)
    k = len(th)
    r2adj = np.nan if (n-k-1)<=0 else 1 - (1-r2)*(n-1)/(n-k-1)

    aic = n*np.log(sse/n) + 2*k
    bic = n*np.log(sse/n) + k*np.log(n)
    aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf

    return th, {"SSE":sse,"R2":r2,"R2_adj":r2adj,"AIC":aic,"AICc":aicc,"BIC":bic}, y_pred


# ------------------------------------------------------------
# 6. Monte Carlo simulation (with optional ROUT)
# ------------------------------------------------------------

def monte_carlo(func, ygen, t_sim, p_true, r_true, lam_true,
                dev_min, dev_max, n_rep, n_points, n_tests,
                y_i, y_f, use_rout, n_phases):

    records = []
    all_y_collected = []   # for global mean/std

    for test in range(1,n_tests+1):

        # ---------- Simulate all replicates ----------
        t_all=[]
        y_all=[]

        for rep in range(n_rep):
            U = np.random.rand(n_points)
            scales = dev_min + (dev_max-dev_min)*U
            noise = scales * np.random.normal(0,1,n_points)
            y_obs = ygen + noise

            t_all.append(t_sim)
            y_all.append(y_obs)

            # Save for global mean/std plotting
            all_y_collected.append(y_obs)

        t_all = np.concatenate(t_all)
        y_all = np.concatenate(y_all)

        # ---------- FIT with or without ROUT ----------
        if use_rout:
            # pre-fit
            th_pre, met_pre, y_pred_pre = fit_polyauxic(t_all, y_all, func, n_phases)
            mask_out = detect_outliers(y_all, y_pred_pre)
            t_clean = t_all[~mask_out]
            y_clean = y_all[~mask_out]
            th, metrics, _ = fit_polyauxic(t_clean, y_clean, func, n_phases)
        else:
            th, metrics, _ = fit_polyauxic(t_all, y_all, func, n_phases)

        # ---------- Parameter extraction ----------
        yi_hat = th[0]
        yf_hat = th[1]
        z = th[2:2+n_phases]
        r = th[2+n_phases:2+2*n_phases]
        lam = th[2+2*n_phases:2+3*n_phases]

        z_shift = z-np.max(z)
        p_hat = np.exp(z_shift)/np.sum(np.exp(z_shift))

        row = {
            "test":test, "yi_hat":yi_hat, "yf_hat":yf_hat,
            "SSE":metrics["SSE"],"R2":metrics["R2"],"R2_adj":metrics["R2_adj"],
            "AIC":metrics["AIC"],"AICc":metrics["AICc"],"BIC":metrics["BIC"]
        }
        for j in range(n_phases):
            row[f"p{j+1}"]=p_hat[j]
            row[f"r{j+1}"]=r[j]
            row[f"lam{j+1}"]=lam[j]

        records.append(row)

    df = pd.DataFrame(records)

    # ---------- GLOBAL MEAN & STD ----------
    all_y_arr = np.vstack(all_y_collected)   # shape (n_tests*n_rep, n_points)
    y_mean = np.mean(all_y_arr, axis=0)
    y_std = np.std(all_y_arr, axis=0)

    return df, y_mean, y_std


# ------------------------------------------------------------
# 7. Streamlit UI
# ------------------------------------------------------------

st.title("Polyauxic Robustness Simulator")

with st.expander("Instructions"):
    st.markdown("""
This Monte Carlo engine tests the robustness of polyauxic kinetics.

Each test:
1. Generates noisy data from the generating function.
2. If enabled, applies ROUT-like outlier removal:
   - Preliminary fit → detect outliers → remove → final fit.
3. Stores the fitted parameters and information criteria.

The second plot shows the generating function and the global mean ± std of all simulated data.
""")

# Sidebar
model = st.sidebar.selectbox("Model",["Boltzmann (Eq 31)", "Gompertz (Eq 32)"])
func = boltzmann_term if "Boltzmann" in model else gompertz_term

n_phases = st.sidebar.number_input("Number of phases",1,10,2)

st.sidebar.subheader("Global Parameters")
y_i = st.sidebar.number_input("y_i",value=0.0)
y_f = st.sidebar.number_input("y_f",value=1.0)

p_true=[]; r_true=[]; lam_true=[]
st.sidebar.subheader("Phase parameters")
for j in range(n_phases):
    with st.sidebar.expander(f"Phase {j+1}",expanded=(j<2)):
        p = st.number_input(f"p{j+1}",min_value=0.0,value=float(1/n_phases))
        r = st.number_input(f"r_max{j+1}",value=1.0)
        lam = st.number_input(f"lambda{j+1}",value=float(j+1))
        p_true.append(p); r_true.append(r); lam_true.append(lam)

st.sidebar.subheader("Noise settings")
dev_min = st.sidebar.number_input("Absolute deviation min",min_value=0.0,value=0.0)
dev_max = st.sidebar.number_input("Absolute deviation max",min_value=0.0,value=0.1)

st.sidebar.subheader("Monte Carlo settings")
n_rep = st.sidebar.number_input("Replicates",1,5,3)
n_points = st.sidebar.number_input("Points per replicate",5,200,50)
n_tests = st.sidebar.number_input("Number of tests",1,200,20)

use_rout = st.sidebar.checkbox("Use ROUT outlier removal?", value=False)

run = st.sidebar.button("Run Analysis")

# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------

if run:

    max_lam = max(lam_true)
    tmax = max(3*max_lam, 1.0)
    t_sim = np.linspace(0,tmax,n_points)

    ygen = polyauxic_generate(t_sim,y_i,y_f,p_true,r_true,lam_true,func)

    # ---- FIRST PLOT: generating function ----
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Generating Function (noise-free)")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(t_sim,ygen,'k-',lw=2,label="Generating function")
        ax.grid(True,ls=":")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        st.pyplot(fig)

    # ---- Run Monte Carlo ----
    df, y_mean, y_std = monte_carlo(
        func, ygen, t_sim, p_true, r_true, lam_true,
        dev_min, dev_max, n_rep, n_points, n_tests,
        y_i, y_f, use_rout, n_phases
    )

    # ---- SECOND PLOT: global mean ± std ----
    with col2:
        st.subheader("Global Mean ± Std of All Simulated Data")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(t_sim, ygen, 'k--', lw=2, label="Generating function")
        ax.errorbar(t_sim, y_mean, yerr=y_std,
                    fmt='o', color='blue', ecolor='gray', capsize=3,
                    label="Mean ± Std")
        ax.grid(True, ls=":")
        ax.legend()
        ax.set_xlabel("t")
        st.pyplot(fig)

    # ---- Table ----
    st.subheader("Monte Carlo Results")
    st.dataframe(df)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="monte_carlo_results.csv",
        mime="text/csv"
    )

    # ---- AIC plots ----
    st.subheader("Information Criteria vs Test")
    fig,ax = plt.subplots(figsize=(8,4))
    for col in ["AIC","AICc","BIC"]:
        ax.plot(df["test"],df[col],marker='o',label=col)
    ax.legend(); ax.grid(True,ls=':')
    ax.set_xlabel("Test")
    st.pyplot(fig)

    # ---- R² plots ----
    st.subheader("R² and Adjusted R² vs Test")
    fig,ax = plt.subplots(figsize=(8,4))
    ax.plot(df["test"],df["R2"],marker='o',label="R²")
    ax.plot(df["test"],df["R2_adj"],marker='s',label="R²_adj")
    ax.legend(); ax.grid(True,ls=':')
    ax.set_xlabel("Test")
    st.pyplot(fig)

    # ---- Parameter plots (2×2) ----
    st.subheader("Parameter Behavior Across Tests")

    fig,axs = plt.subplots(2,2,figsize=(12,8))

    # yi & yf
    axs[0,0].plot(df["test"],df["yi_hat"],label="yi_hat")
    axs[0,0].plot(df["test"],df["yf_hat"],label="yf_hat")
    axs[0,0].set_title("y_i and y_f")
    axs[0,0].legend(); axs[0,0].grid(True,ls=':')

    # p_j
    for j in range(n_phases):
        axs[0,1].plot(df["test"],df[f"p{j+1}"],label=f"p{j+1}")
    axs[0,1].set_title("p_j")
    axs[0,1].legend(); axs[0,1].grid(True,ls=':')

    # r_j
    for j in range(n_phases):
        axs[1,0].plot(df["test"],df[f"r{j+1}"],label=f"r_max{j+1}")
    axs[1,0].set_title("r_max_j")
    axs[1,0].legend(); axs[1,0].grid(True,ls=':')

    # lambda_j
    for j in range(n_phases):
        axs[1,1].plot(df["test"],df[f"lam{j+1}"],label=f"lambda{j+1}")
    axs[1,1].set_title("lambda_j")
    axs[1,1].legend(); axs[1,1].grid(True,ls=':')

    st.pyplot(fig)
