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

Polyauxic Robustness Simulator
================================

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

2) A **generating function** (noise-free curve) is computed once:
       y_gen(t) = y_i + (y_f - y_i) * sum_j term_j(t)

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
   - Graph showing the generating function
   - Graph showing the generating function + one noisy replicate
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


# ------------------------------------------------------------
# 1. Model Equations (Boltzmann Eq. 31 and Gompertz Eq. 32)
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
# 2. Polyauxic generating function (noise-free)
# ------------------------------------------------------------

def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    p_vec = np.asarray(p_vec)
    # normalize p
    if np.sum(p_vec) <= 0:
        p_vec = np.ones_like(p_vec) / len(p_vec)
    else:
        p_vec = p_vec / np.sum(p_vec)
    sum_terms = np.zeros_like(t, dtype=float)
    for j in range(len(p_vec)):
        sum_terms += func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
    return y_i + (y_f - y_i) * sum_terms


# ------------------------------------------------------------
# 3. Polyauxic model for fitting (softmax parametrization)
# ------------------------------------------------------------

def polyauxic_fit_model(t, theta, func, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r = theta[2 + n_phases : 2 + 2*n_phases]
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]

    # softmax
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    p = expz / np.sum(expz)

    sum_terms = np.zeros_like(t)
    for j in range(n_phases):
        sum_terms += func(t, y_i, y_f, p[j], r[j], lam[j])

    return y_i + (y_f - y_i) * sum_terms


def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1*np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred)**2)


# ------------------------------------------------------------
# 4. Smart initial guess (simplified)
# ------------------------------------------------------------

def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    if len(dy) >= 5:
        dy_s = np.convolve(dy, np.ones(5)/5, mode='same')
    else:
        dy_s = dy.copy()

    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guesses = []
    if len(peaks) > 0:
        idx = np.argsort(props['peak_heights'])[::-1][:n_phases]
        best = peaks[idx]
        for p in best:
            guesses.append((t[p], abs(dy_s[p])))
    while len(guesses) < n_phases:
        tspan = t.max() - t.min() if t.max() > t.min() else 1
        guesses.append((t.min() + tspan*(len(guesses)+1)/(n_phases+1),
                        (y.max()-y.min())/(tspan/n_phases)))

    guesses = sorted(guesses, key=lambda x: x[0])

    theta0 = np.zeros(2 + 3*n_phases)
    theta0[0] = y.min()
    theta0[1] = y.max()
    for i,(lam,r) in enumerate(guesses):
        theta0[2 + n_phases + i] = r
        theta0[2 + 2*n_phases + i] = lam
    return theta0


# ------------------------------------------------------------
# 5. Fitting engine (DE → L-BFGS-B)
# ------------------------------------------------------------

def fit_polyauxic(t_all, y_all, func, n_phases):
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1
    t_n = t_all / t_scale
    y_n = y_all / y_scale

    theta0 = smart_guess(t_all, y_all, n_phases)
    # normalize
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0] / y_scale
    th0[1] = theta0[1] / y_scale
    th0[2:2+n_phases] = 0
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases] * t_scale / y_scale
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases] / t_scale

    bounds = [(-0.2,1.5),(0,2)]
    bounds += [(-10,10)]*n_phases
    bounds += [(0,500)]*n_phases
    bounds += [(-0.1,1.2)]*n_phases

    popsize=20
    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))

    res_de = differential_evolution(
        sse_loss, bounds,
        args=(t_n,y_n,func,n_phases),
        maxiter=800, popsize=popsize, init=init_pop,
        strategy="best1bin", polish=True, tol=1e-6
    )
    res = minimize(
        sse_loss, res_de.x,
        args=(t_n,y_n,func,n_phases),
        method="L-BFGS-B", bounds=bounds, tol=1e-10
    )

    th_n = res.x
    # map back
    th = np.zeros_like(th_n)
    th[0] = th_n[0]*y_scale
    th[1] = th_n[1]*y_scale
    th[2:2+n_phases] = th_n[2:2+n_phases]
    th[2+n_phases:2+2*n_phases] = th_n[2+n_phases:2+2*n_phases]*(y_scale/t_scale)
    th[2+2*n_phases:2+3*n_phases] = th_n[2+2*n_phases:2+3*n_phases]*t_scale

    y_pred = polyauxic_fit_model(t_all, th, func, n_phases)
    sse = np.sum((y_all - y_pred)**2)
    sst = np.sum((y_all - np.mean(y_all))**2)
    r2 = 1 - sse/sst if sst>0 else np.nan

    n = len(y_all)
    k = len(th)
    r2adj = np.nan if (n-k-1)<=0 else 1 - (1-r2)*(n-1)/(n-k-1)
    aic = n*np.log(sse/n) + 2*k
    bic = n*np.log(sse/n) + k*np.log(n)
    aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf

    return th, {"SSE":sse,"R2":r2,"R2_adj":r2adj,"AIC":aic,"AICc":aicc,"BIC":bic}


# ------------------------------------------------------------
# 6. Monte Carlo Simulation
# ------------------------------------------------------------

def monte_carlo(func, ygen, t_sim, p_true, r_true, lam_true,
                dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f):

    results = []
    n_phases = len(p_true)

    for test in range(1, n_tests+1):
        t_all = []
        y_all = []

        for rep in range(n_rep):
            U = np.random.rand(n_points)
            scales = dev_min + (dev_max-dev_min)*U
            noise = scales * np.random.normal(0,1,n_points)
            y_obs = ygen + noise

            t_all.append(t_sim)
            y_all.append(y_obs)

        t_all = np.concatenate(t_all)
        y_all = np.concatenate(y_all)

        th, met = fit_polyauxic(t_all, y_all, func, n_phases)

        yi_hat = th[0]
        yf_hat = th[1]
        z = th[2:2+n_phases]
        r = th[2+n_phases:2+2*n_phases]
        lam = th[2+2*n_phases:2+3*n_phases]

        z_shift = z-np.max(z)
        p_hat = np.exp(z_shift)/np.sum(np.exp(z_shift))

        row = {
            "test":test, "yi_hat":yi_hat, "yf_hat":yf_hat,
            "SSE":met["SSE"],"R2":met["R2"],"R2_adj":met["R2_adj"],
            "AIC":met["AIC"],"AICc":met["AICc"],"BIC":met["BIC"]
        }
        for j in range(n_phases):
            row[f"p{j+1}"]=p_hat[j]
            row[f"r{j+1}"]=r[j]
            row[f"lam{j+1}"]=lam[j]

        results.append(row)

    return pd.DataFrame(results)


# ------------------------------------------------------------
# 7. Streamlit App
# ------------------------------------------------------------

st.title("Polyauxic Robustness Simulator")

with st.expander("Instructions"):
    st.markdown("""
This tool performs Monte Carlo robustness testing of polyauxic kinetic models.

**Steps executed:**
1. A *generating function* (noise-free curve) is computed from the true parameters.
2. For each Monte Carlo test, new independent noise is added.
3. All replicates are concatenated and fitted using Differential Evolution + L-BFGS-B.
4. Fitted parameters and information criteria are recorded.

**Outputs:**
- Generating function plot
- Example noisy dataset plot
- Table of Monte Carlo results
- AIC/AICc/BIC behavior
- R2 / R2_adj behavior
- 4 parameter plots in a 2×2 layout:
  - yi & yf
  - p_j
  - r_max_j
  - lambda_j
""")

# Sidebar selections
model = st.sidebar.selectbox("Model",["Boltzmann (Eq 31)","Gompertz (Eq 32)"])
func = boltzmann_term if "Boltzmann" in model else gompertz_term

n_phases = st.sidebar.number_input("Number of phases",1,10,2)

st.sidebar.subheader("Global Parameters")
y_i = st.sidebar.number_input("y_i",value=0.0)
y_f = st.sidebar.number_input("y_f",value=1.0)

p_true=[]; r_true=[]; lam_true=[]
st.sidebar.subheader("Phase Parameters")
for j in range(n_phases):
    with st.sidebar.expander(f"Phase {j+1}",expanded=(j<2)):
        p = st.number_input(f"p{j+1}",min_value=0.0,value=float(1/n_phases))
        r = st.number_input(f"r_max{j+1}",value=1.0)
        lam = st.number_input(f"lambda{j+1}",value=float(j+1))
        p_true.append(p)
        r_true.append(r)
        lam_true.append(lam)

st.sidebar.subheader("Noise Settings")
dev_min = st.sidebar.number_input("Absolute deviation min",min_value=0.0,value=0.0)
dev_max = st.sidebar.number_input("Absolute deviation max",min_value=0.0,value=0.1)

st.sidebar.subheader("Monte Carlo Settings")
n_rep = st.sidebar.number_input("Replicates",1,5,3)
n_points = st.sidebar.number_input("Points per replicate",5,200,50)
n_tests = st.sidebar.number_input("Number of tests",1,200,20)

run = st.sidebar.button("Run Analysis")


# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------

if run:
    # generating function t range
    max_lam = max(lam_true)
    tmax = max(3*max_lam,1.0)
    t_sim = np.linspace(0,tmax,n_points)

    ygen = polyauxic_generate(t_sim,y_i,y_f,p_true,r_true,lam_true,func)

    # pick a random test to show noisy sample
    test_idx = np.random.randint(1,n_tests+1)
    U = np.random.rand(n_points)
    scales = dev_min + (dev_max-dev_min)*U
    noise = scales*np.random.normal(0,1,n_points)
    noisy_example = ygen + noise

    # ------------------------------------------------------------
    # GRAPH 1 – generating function
    # GRAPH 2 – one noisy dataset
    # ------------------------------------------------------------
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Generating Function (Noise-free)")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(t_sim,ygen,'k-',lw=2,label="y_gen(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.grid(True,ls=':')
        st.pyplot(fig)

    with col2:
        st.subheader("Example Noisy Dataset (1 replicate)")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(t_sim,ygen,'k--',lw=1,label="generating function")
        ax.scatter(t_sim,noisy_example,color='red',s=20,label="noisy data")
        ax.legend()
        ax.grid(True,ls=':')
        st.pyplot(fig)

    # ------------------------------------------------------------
    # Monte Carlo execution
    # ------------------------------------------------------------
    df = monte_carlo(func,ygen,t_sim,p_true,r_true,lam_true,
                     dev_min,dev_max,n_rep,n_points,n_tests,y_i,y_f)

    st.subheader("Monte Carlo Results Table")
    st.dataframe(df)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="monte_carlo_results.csv",
        mime="text/csv"
    )

    # ------------------------------------------------------------
    # AIC / AICc / BIC
    # ------------------------------------------------------------
    st.subheader("Information Criteria vs Test")
    fig,ax = plt.subplots(figsize=(8,4))
    for col in ["AIC","AICc","BIC"]:
        ax.plot(df["test"],df[col],marker='o',label=col)
    ax.legend(); ax.grid(True,ls=':')
    ax.set_xlabel("Test"); ax.set_ylabel("Criterion value")
    st.pyplot(fig)

    # ------------------------------------------------------------
    # R2 and R2_adj
    # ------------------------------------------------------------
    st.subheader("R² and Adjusted R² vs Test")
    fig,ax = plt.subplots(figsize=(8,4))
    ax.plot(df["test"],df["R2"],marker='o',label="R²")
    ax.plot(df["test"],df["R2_adj"],marker='s',label="R²_adj")
    ax.legend(); ax.grid(True,ls=':')
    ax.set_xlabel("Test"); ax.set_ylabel("Value")
    st.pyplot(fig)

    # ------------------------------------------------------------
    # PARAMETER PLOTS (2 × 2 layout)
    # ------------------------------------------------------------
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
