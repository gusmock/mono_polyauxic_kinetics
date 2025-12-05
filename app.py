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

DESCRIPTION
-----------
This Streamlit application performs *Monte Carlo robustness testing* for
polyauxic kinetic models, specifically the Boltzmann (Eq. 31) and Gompertz
(Eq. 32) formulations described in the original mono/polyauxic kinetics paper.

The purpose of this tool is to verify the *stability*, *identifiability* and
*sensitivity* of a polyauxic kinetic model under realistic experimental noise.

The application allows the user to impose the “true” kinetic parameters and
then observe how random fluctuations in experimental measurements affect the
quality of the estimated parameters and information criteria.

--------------------------------------------------------------------------------
FULL ALGORITHM SUMMARY
--------------------------------------------------------------------------------

1) USER INPUTS TRUE PARAMETERS
   The user sets:
       • Model type (Boltzmann or Gompertz)
       • Number of phases n
       • Global parameters:
             y_i < y_f
       • Phase parameters for each j = 1..n:
             p_j > 0   AND   sum_j p_j = 1   (automatically normalized)
             r_max_j > 0
             lambda_j strictly increasing: λ1 < λ2 < ... < λn

   PARAMETER VALIDATION is performed before running any simulation.
   Invalid configurations immediately stop the execution.

2) GENERATING FUNCTION (NOISE-FREE)
   Using the true parameters, the algorithm creates:
       y_gen(t) = y_i + (y_f - y_i) * Σ_j f_j(t)
   where f_j(t) is:
       • Boltzmann term (Eq. 31), or
       • Gompertz term (Eq. 32)
   depending on the selected model.

   This is the "perfect curve" the experiment SHOULD produce in absence of
   measurement noise. It is displayed in the first plot.

3) NOISE MODEL (MATCHES EXCEL BEHAVIOR EXACTLY)
   Noise is applied following your Excel-like formulation:

       noise = scale * Normal(0, 1)
       scale = dev_min + (dev_max - dev_min) * Uniform(0,1)

   This ensures the amplitude of the noise varies randomly, just like in your
   spreadsheet implementation.

4) SIMULATION OF EXPERIMENTAL DATA
   For EACH Monte Carlo test:
       For EACH replicate:
           • Generate new noise independently
           • Apply noise to the generating function:
                 y_obs = y_gen + noise
           • Store the replicate

   All replicates are concatenated and form a simulated experimental dataset.

5) OPTIONAL ROUT OUTLIER REMOVAL (LIKE YOUR MAIN APP)
   If enabled:
       a) Preliminary fit is performed
       b) Residuals are computed
       c) Outliers are detected using MAD-based ROUT criterion:
            z = |residual - median| / (1.4826 * MAD)
            outlier if z > 2.5
       d) Outliers are removed
       e) Final fit is performed only on the clean data

   If ROUT is disabled:
       • Only one fit is performed using all data.

6) FITTING ENGINE (IDENTICAL METHODOLOGY)
       Differential Evolution → L-BFGS-B refinement
   All parameter transformations and softmax normalization for p_j are preserved.

7) METRICS STORED FOR EACH TEST
       • Fitted parameters: y_i_hat, y_f_hat, p_j_hat, r_j_hat, lambda_j_hat
       • SSE
       • R²
       • Adjusted R²
       • AIC
       • AICc
       • BIC

8) GLOBAL STATISTICS ACROSS ALL TESTS
   The second plot shows:
       • The generating function
       • The MEAN of ALL simulated values across ALL tests and replicates
       • The STANDARD DEVIATION across all datasets

   This gives a clear picture of how much noise deviates the observed mean
   response from the ideal kinetic curve.

9) PARAMETER ROBUSTNESS
   The application produces:
       • AIC/AICc/BIC curves across tests
       • R² and Adjusted R² curves across tests
       • A 2×2 grid with:
             (A) y_i & y_f behavior
             (B) p_j behavior
             (C) r_max_j behavior
             (D) lambda_j behavior

   These plots reveal convergence, identifiability problems, phase overlap,
   and parameter sensitivity under noise.

================================================================================
END OF OVERVIEW
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor  # PARALLEL MC


# ------------------------------------------------------------
# ROUT OUTLIER DETECTION
# ------------------------------------------------------------
def detect_outliers(y_true, y_pred):
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5


# ------------------------------------------------------------
# MODEL EQUATIONS
# ------------------------------------------------------------
def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    expo = (4*r_j*(lam_j - t)) / (dy*p_j) + 2
    expo = np.clip(expo, -500, 500)
    return p_j / (1 + np.exp(expo))


def gompertz_term(t, y_i, y_f, p_j, r_j, lam_j):
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    expo = (r_j*np.e*(lam_j - t)) / (dy*p_j) + 1
    expo = np.clip(expo, -500, 500)
    return p_j * np.exp(-np.exp(expo))


# ------------------------------------------------------------
# GENERATING FUNCTION
# ------------------------------------------------------------
def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    p_norm = np.array(p_vec) / np.sum(p_vec)
    y_out = np.zeros_like(t)
    for j in range(len(p_norm)):
        y_out += func(t, y_i, y_f, p_norm[j], r_vec[j], lam_vec[j])
    return y_i + (y_f - y_i)*y_out


# ------------------------------------------------------------
# FITTING MODEL (softmax)
# ------------------------------------------------------------
def polyauxic_fit_model(t, theta, func, n_phases):
    y_i, y_f = theta[0], theta[1]
    z = theta[2:2+n_phases]
    r = theta[2+n_phases:2+2*n_phases]
    lam = theta[2+2*n_phases:2+3*n_phases]

    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))

    y_out = np.zeros_like(t)
    for j in range(n_phases):
        y_out += func(t, y_i, y_f, p[j], r[j], lam[j])
    return y_i + (y_f - y_i)*y_out


def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    return np.sum((y - y_pred)**2)


# ------------------------------------------------------------
# INITIAL GUESS
# ------------------------------------------------------------
def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    try:
        dy_s = np.convolve(dy, np.ones(5)/5, mode="same")
    except:
        dy_s = dy.copy()

    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guess=[]
    if len(peaks)>0:
        idx = np.argsort(props["peak_heights"])[::-1][:n_phases]
        for p in peaks[idx]:
            guess.append((t[p], abs(dy_s[p])))

    while len(guess)<n_phases:
        span = t.max()-t.min() if t.max()>t.min() else 1
        guess.append((t.min()+span*(len(guess)+1)/(n_phases+1),
                      (y.max()-y.min())/(span/n_phases)))

    theta0 = np.zeros(2+3*n_phases)
    theta0[0]=y.min()
    theta0[1]=y.max()
    for i,(lam_,r_) in enumerate(guess):
        theta0[2+n_phases+i]=r_
        theta0[2+2*n_phases+i]=lam_
    return theta0


# ------------------------------------------------------------
# FITTING ENGINE
# ------------------------------------------------------------
def fit_polyauxic(t_all, y_all, func, n_phases):
    t_scale = max(np.max(t_all), 1e-9)
    y_scale = max(np.max(np.abs(y_all)), 1e-9)

    t_n = t_all/t_scale
    y_n = y_all/y_scale

    theta0 = smart_guess(t_all, y_all, n_phases)

    # Normalize
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale
    th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases]*t_scale/y_scale
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases]/t_scale

    bounds=[(-0.2,1.5),(0,2)]
    bounds+= [(-10,10)]*n_phases
    bounds+= [(0,500)]*n_phases
    bounds+= [(-0.1,1.2)]*n_phases

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
# SINGLE TEST (for parallel MC)
# ------------------------------------------------------------
def run_single_test(args):
    (
        test_id, func, ygen, t_sim,
        p_true, r_true, lam_true,
        dev_min, dev_max,
        n_rep, n_points,
        use_rout, n_phases
    ) = args

    all_reps = []
    t_all = []
    y_all = []

    for _ in range(n_rep):
        U = np.random.rand(n_points)
        scales = dev_min + (dev_max-dev_min)*U
        noise = scales * np.random.normal(0,1,n_points)
        y_obs = ygen + noise

        t_all.append(t_sim)
        y_all.append(y_obs)
        all_reps.append(y_obs)

    t_all = np.concatenate(t_all)
    y_all = np.concatenate(y_all)

    if use_rout:
        th_pre, _, y_pred_pre = fit_polyauxic(t_all, y_all, func, n_phases)
        out_mask = detect_outliers(y_all, y_pred_pre)
        t_clean = t_all[~out_mask]
        y_clean = y_all[~out_mask]
        th, metrics, _ = fit_polyauxic(t_clean, y_clean, func, n_phases)
    else:
        th, metrics, _ = fit_polyauxic(t_all, y_all, func, n_phases)

    yi_hat, yf_hat = th[0], th[1]
    z = th[2:2+n_phases]
    r = th[2+n_phases:2+2*n_phases]
    lam = th[2+2*n_phases:2+3*n_phases]

    p_hat = np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)))

    row={"test":test_id,"yi_hat":yi_hat,"yf_hat":yf_hat,
         "SSE":metrics["SSE"],"R2":metrics["R2"],"R2_adj":metrics["R2_adj"],
         "AIC":metrics["AIC"],"AICc":metrics["AICc"],"BIC":metrics["BIC"]}

    for j in range(n_phases):
        row[f"p{j+1}"]=p_hat[j]
        row[f"r{j+1}"]=r[j]
        row[f"lam{j+1}"]=lam[j]

    return row, np.vstack(all_reps)


# ------------------------------------------------------------
# PARALLEL MONTE CARLO
# ------------------------------------------------------------
def monte_carlo_parallel(
    func, ygen, t_sim, p_true, r_true, lam_true,
    dev_min, dev_max, n_rep, n_points, n_tests,
    use_rout, n_phases,
    progress_bar, status_text
):
    tasks = []
    for test in range(1, n_tests+1):
        tasks.append((
            test, func, ygen, t_sim,
            p_true, r_true, lam_true,
            dev_min, dev_max,
            n_rep, n_points,
            use_rout, n_phases
        ))

    results = []
    all_y = []

    with ProcessPoolExecutor() as ex:
        futures = ex.map(run_single_test, tasks)
        for k, (row, rep_block) in enumerate(futures):
            results.append(row)
            all_y.append(rep_block)
            if progress_bar is not None:
                progress_bar.progress((k+1)/n_tests)
            if status_text is not None:
                status_text.text(f"Running test {k+1}/{n_tests}...")

    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.text("Done.")

    df = pd.DataFrame(results)
    all_y_arr = np.vstack(all_y)
    y_mean = np.mean(all_y_arr,axis=0)
    y_std = np.std(all_y_arr,axis=0)

    return df, y_mean, y_std


# ------------------------------------------------------------
# STREAMLIT UI (main app)
# ------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Polyauxic Robustness Simulator")

    with st.expander("Instructions"):
        st.markdown("""
### What this application does  
This tool simulates *synthetic experimental datasets* using a noise-free 
polyauxic kinetic curve (the generating function) and repeatedly perturbs
it with stochastic noise to analyze parameter robustness.

### Steps performed internally
1. Construct the generating function from your parameters.  
2. Simulate noise EXACTLY like your Excel formula.  
3. Generate N Monte Carlo test datasets.  
4. Fit each dataset using the same DE→L-BFGS-B optimizer.  
5. (Optional) Apply ROUT-based outlier removal.  
6. Record parameters, R², AIC, BIC, etc.  
7. Plot robustness of every kinetic parameter.  
8. Show global deviation (mean ± std) of all experiments.

### Parameter restrictions enforced
- `y_i < y_f`  
- all `p_j > 0` and renormalized so Σ p_j = 1  
- `lambda_1 < lambda_2 < ... < lambda_n`  

If any rule is violated, the analysis aborts with an error.
""")

    # Sidebar inputs --------------------------------------------------------------
    model = st.sidebar.selectbox("Model",["Boltzmann (Eq. 31)", "Gompertz (Eq. 32)"])
    func = boltzmann_term if "Boltzmann" in model else gompertz_term

    n_phases = st.sidebar.number_input("Number of phases",1,10,2)

    st.sidebar.subheader("Global parameters")
    y_i = st.sidebar.number_input("y_i",value=0.0)
    y_f = st.sidebar.number_input("y_f",value=1.0)

    p_true=[]; r_true=[]; lam_true=[]

    st.sidebar.subheader("Phase parameters")
    for j in range(n_phases):
        with st.sidebar.expander(f"Phase {j+1}", expanded=(j<2)):
            p = st.number_input(f"p{j+1}",min_value=0.0,value=float(1/n_phases))
            r = st.number_input(f"r_max{j+1}",value=1.0)
            lam = st.number_input(f"lambda{j+1}",value=float(j+1))
            p_true.append(p); r_true.append(r); lam_true.append(lam)

    st.sidebar.subheader("Noise settings")
    dev_min = st.sidebar.number_input("Absolute deviation min",min_value=0.0,value=0.0)
    dev_max = st.sidebar.number_input("Absolute deviation max",min_value=0.0,value=0.1)

    st.sidebar.subheader("Monte Carlo settings")
    n_rep = st.sidebar.number_input("Replicates",1,5,3)
    n_points = st.sidebar.number_input("Points per replicate",5,250,50)
    n_tests = st.sidebar.number_input("Number of tests",1,250,20)

    use_rout = st.sidebar.checkbox("Use ROUT outlier removal?",value=False)

    run = st.sidebar.button("Run Analysis")

    # ---------------------------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------------------------
    if run:

        # 1. PARAMETER VALIDATION ---------------------------------------------
        if y_i >= y_f:
            st.error("Invalid parameters: y_i must be strictly less than y_f.")
            st.stop()

        p_arr = np.array(p_true, dtype=float)
        if (p_arr<=0).any():
            st.error("All p_j must be > 0.")
            st.stop()
        p_arr = p_arr / np.sum(p_arr)
        p_true_norm = p_arr.tolist()

        lam_arr = np.array(lam_true, dtype=float)
        if not np.all(np.diff(lam_arr)>0):
            st.error("lambda_j must satisfy λ1 < λ2 < ... < λn.")
            st.stop()
        lam_true_sorted = lam_arr.tolist()

        # 2. GENERATING FUNCTION ----------------------------------------------
        max_lam = max(lam_true_sorted)
        tmax = max(3*max_lam, 1.0)
        t_sim = np.linspace(0,tmax,n_points)

        ygen = polyauxic_generate(t_sim,y_i,y_f,p_true_norm,r_true,lam_true_sorted,func)

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Generating Function (Noise-free)")
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(t_sim, ygen, "k-", lw=2)
            ax.set_xlabel("t")
            ax.set_ylabel("y")
            dy = abs(y_f - y_i)
            ax.set_ylim(min(y_i,y_f)-0.05*dy, max(y_i,y_f)+0.05*dy)
            ax.grid(True, ls=":")
            st.pyplot(fig)

        # 3. MONTE CARLO (PARALLEL + PROGRESS BAR) ---------------------------
        progress_bar = st.progress(0)
        status_text = st.empty()

        df, y_mean, y_std = monte_carlo_parallel(
            func, ygen, t_sim,
            p_true_norm, r_true, lam_true_sorted,
            dev_min, dev_max,
            n_rep, n_points, n_tests,
            use_rout, n_phases,
            progress_bar, status_text
        )

        # 4. MEAN ± STD PLOT ---------------------------------------------------
        with col2:
            st.subheader("Global Mean ± Std of All Simulated Data")
            fig2,ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(t_sim, ygen, "k--", lw=2, label="Generating function")
            ax2.errorbar(t_sim, y_mean, yerr=y_std,
                        fmt="o", color="blue", ecolor="gray",
                        capsize=3, label="Mean ± Std")
            ax2.set_ylim(min(y_i,y_f)-0.05*dy, max(y_i,y_f)+0.05*dy)
            ax2.grid(True, ls=":")
            ax2.legend()
            st.pyplot(fig2)

        # 5. TABLE --------------------------------------------------------------
        st.subheader("Monte Carlo Results")
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="monte_carlo_results.csv",
            mime="text/csv"
        )

        # 6. INFORMATION CRITERIA ----------------------------------------------
        st.subheader("Information Criteria vs Test")
        fig3,ax3 = plt.subplots(figsize=(8,4))
        for col in ["AIC","AICc","BIC"]:
            ax3.plot(df["test"],df[col],marker="o",label=col)
        ax3.legend()
        ax3.grid(True,ls=":")
        st.pyplot(fig3)

        # 7. R2 PLOTS -----------------------------------------------------------
        st.subheader("R² and Adjusted R² vs Test")
        fig4,ax4 = plt.subplots(figsize=(8,4))
        ax4.plot(df["test"],df["R2"],marker="o",label="R²")
        ax4.plot(df["test"],df["R2_adj"],marker="s",label="R²_adj")
        ax4.legend()
        ax4.set_xlabel("Test")
        ax4.grid(True,ls=":")
        st.pyplot(fig4)

        # 8. PARAMETER PLOTS (2×2 layout) --------------------------------------
        st.subheader("Parameter Behavior Across Tests")
        fig5,axs = plt.subplots(2,2,figsize=(12,8))

        # yi,yf
        axs[0,0].plot(df["test"],df["yi_hat"],label="yi_hat")
        axs[0,0].plot(df["test"],df["yf_hat"],label="yf_hat")
        axs[0,0].set_title("y_i and y_f")
        axs[0,0].grid(True, ls=":")
        axs[0,0].legend()

        # p_j
        for j in range(n_phases):
            axs[0,1].plot(df["test"],df[f"p{j+1}"],label=f"p{j+1}")
        axs[0,1].set_title("p_j")
        axs[0,1].grid(True, ls=":")
        axs[0,1].legend()

        # r_j
        for j in range(n_phases):
            axs[1,0].plot(df["test"],df[f"r{j+1}"],label=f"r_max{j+1}")
        axs[1,0].set_title("r_max_j")
        axs[1,0].grid(True, ls=":")
        axs[1,0].legend()

        # lambda_j
        for j in range(n_phases):
            axs[1,1].plot(df["test"],df[f"lam{j+1}"],label=f"lambda{j+1}")
        axs[1,1].set_title("lambda_j")
        axs[1,1].grid(True, ls=":")
        axs[1,1].legend()

        st.pyplot(fig5)


if __name__ == "__main__":
    main()
