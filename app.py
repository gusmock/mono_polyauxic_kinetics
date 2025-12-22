"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 

-------------------------------------------------------------------------------
POLYAUXIC MODELING PLATFORM: COMPUTATIONAL WORKFLOW & THEORETICAL FRAMEWORK
-------------------------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import t as t_dist

# -----------------------------------------------------------------------------
# Minimal proof-of-concept: Polyauxic Modeling Platform (English only)
# Core methodology: Eq.31 (Boltzmann), Eq.32 (Gompertz), Eq.33 (Softmax weights),
# DE -> L-BFGS-B, ROUT(FDR) with Q fixed at 1.0%, AIC/AICc/BIC selection.
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "mathtext.fontset": "stix",
})

# =========================
# Eq. 31 and Eq. 32 Terms
# =========================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Eq. 31 term:
    p_j / (1 + exp( (4*r_max_j*(lambda_j - t))/((y_f - y_i)*p_j) + 2 ))
    """
    t = np.asarray(t, dtype=float)
    delta = (y_f - y_i)
    delta = delta if abs(delta) > 1e-12 else 1e-12
    p_safe = max(float(p_j), 1e-12)

    exponent = (4.0 * r_max_j * (lambda_j - t)) / (delta * p_safe) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Eq. 32 term:
    p_j * exp( -exp( (r_max_j*e*(lambda_j - t))/((y_f - y_i)*p_j) + 1 ) )
    """
    t = np.asarray(t, dtype=float)
    delta = (y_f - y_i)
    delta = delta if abs(delta) > 1e-12 else 1e-12
    p_safe = max(float(p_j), 1e-12)

    exponent = (r_max_j * np.e * (lambda_j - t)) / (delta * p_safe) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe * np.exp(-np.exp(exponent))

# =========================
# Eq. 33 (Softmax weights)
# =========================

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)

# =========================
# Polyauxic sum (global model)
# =========================

def polyauxic_model(t, theta, term_func, n_phases):
    """
    theta = [y_i, y_f, z1..zn, r1..rn, l1..ln]
    p = softmax(z)  (Eq. 33)
    y(t) = y_i + (y_f - y_i) * sum_j term_func(...)
    """
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]

    z = theta[2 : 2 + n_phases]
    r = theta[2 + n_phases : 2 + 2*n_phases]
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]

    p = softmax(z)

    s = 0.0
    for j in range(n_phases):
        s += term_func(t, y_i, y_f, p[j], r[j], lam[j])

    return y_i + (y_f - y_i) * s

# =========================
# Loss functions
# =========================

def _lambda_order_penalty(theta, n_phases):
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]
    if np.any(np.diff(lam) <= 0):
        return 1e12
    return 0.0

def sse_loss(theta, t, y, term_func, n_phases):
    pen = _lambda_order_penalty(theta, n_phases)
    if pen > 0:
        return pen
    yhat = polyauxic_model(t, theta, term_func, n_phases)
    return float(np.sum((y - yhat) ** 2))

def soft_l1_loss(theta, t, y, term_func, n_phases):
    """
    Soft L1 / Charbonnier-like robust loss used for robust pre-fit
    (to avoid extreme points dominating the initial fit).
    """
    pen = _lambda_order_penalty(theta, n_phases)
    if pen > 0:
        return pen
    yhat = polyauxic_model(t, theta, term_func, n_phases)
    r = y - yhat
    # Soft L1: 2*(sqrt(1 + r^2) - 1)
    return float(np.sum(2.0 * (np.sqrt(1.0 + r*r) - 1.0)))

# =========================
# ROUT outliers (FDR fixed)
# =========================

def detect_outliers_rout_fdr(y_true, y_pred, Q=1.0):
    """
    ROUT-like with FDR control (Benjamini-Hochberg), using robust scale via MAD.
    Q is fixed at 1.0% in the app (as requested).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = y_true - y_pred
    n = res.size
    if n < 3:
        return np.zeros_like(res, dtype=bool)

    med = np.median(res)
    mad = np.median(np.abs(res - med))
    rsdr = 1.4826 * mad if mad > 1e-12 else 1e-12

    t_scores = res / rsdr
    df = max(n - 1, 1)
    pvals = 2.0 * (1.0 - t_dist.cdf(np.abs(t_scores), df=df))

    alpha = Q / 100.0
    idx = np.argsort(pvals)
    p_sorted = pvals[idx]
    i = np.arange(1, n + 1)
    thresh = (i / n) * alpha
    ok = p_sorted <= thresh
    if not np.any(ok):
        return np.zeros_like(res, dtype=bool)

    kmax = np.max(np.where(ok)[0])
    pcrit = p_sorted[kmax]
    return pvals <= pcrit

# =========================
# Initialization heuristics
# =========================

def smart_initial_guess(t, y, n_phases):
    """
    Peak-based heuristic using derivative peaks as starting points for lambda and r_max.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    dy = np.gradient(y, t)
    dy_sm = np.convolve(dy, np.ones(5) / 5.0, mode="same")

    min_dist = max(1, len(t) // max(1, (n_phases * 4)))
    peaks, props = find_peaks(dy_sm, height=np.max(dy_sm) * 0.1, distance=min_dist)

    guesses = []
    if len(peaks) > 0:
        order = np.argsort(props["peak_heights"])[::-1]
        best = peaks[order][:n_phases]
        for k in best:
            guesses.append({"lambda": float(t[k]), "r_max": float(dy_sm[k])})

    while len(guesses) < n_phases:
        tspan = float(t.max() - t.min()) if float(t.max() - t.min()) > 0 else 1.0
        guesses.append({
            "lambda": float(t.min() + tspan * (len(guesses) + 1) / (n_phases + 1)),
            "r_max": float((np.max(y) - np.min(y)) / max(tspan / n_phases, 1e-9))
        })

    guesses.sort(key=lambda d: d["lambda"])

    theta = np.zeros(2 + 3*n_phases, dtype=float)
    theta[0] = float(np.min(y))
    theta[1] = float(np.max(y))
    theta[2 : 2+n_phases] = 0.0  # z

    for i in range(n_phases):
        theta[2 + n_phases + i] = guesses[i]["r_max"]
        theta[2 + 2*n_phases + i] = guesses[i]["lambda"]

    return theta

# =========================
# Information criteria
# =========================

def choose_information_criterion(N, k_max):
    """
    Same rule used in your platform: decide among AIC/AICc/BIC using N and N/k.
    """
    ratio = N / max(k_max, 1)
    if N <= 200:
        return "AICc" if ratio < 40 else "AIC"
    return "BIC"

def first_local_min_index(values, tol=1e-12):
    """
    First local minimum rule used in your workflow:
    keep increasing phases while criterion improves; stop at the first non-improvement.
    """
    if len(values) == 0:
        return 0
    best = 0
    for i in range(1, len(values)):
        if values[i] < values[best] - tol:
            best = i
        else:
            break
    return best

# =========================
# Fitting engine (DE -> L-BFGS-B)
# =========================

def fit_model(t, y, term_func, n_phases):
    """
    Fit with SSE loss. Uses normalized variables for stability,
    runs Differential Evolution then L-BFGS-B.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n_params = 2 + 3*n_phases
    if len(t) <= n_params + 2:
        return None

    t_scale = float(np.max(t)) if float(np.max(t)) > 0 else 1.0
    y_scale = float(np.max(y)) if float(np.max(y)) > 0 else 1.0
    t_norm = t / t_scale
    y_norm = y / y_scale

    theta0 = smart_initial_guess(t, y, n_phases)
    theta0n = np.zeros_like(theta0)
    theta0n[0] = theta0[0] / y_scale
    theta0n[1] = theta0[1] / y_scale
    theta0n[2:2+n_phases] = 0.0
    theta0n[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases] / (y_scale / t_scale)
    theta0n[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases] / t_scale

    bounds = []
    bounds.append((0.0, 2.0))   # y_i (normalized)
    bounds.append((0.0, 2.0))   # y_f (normalized)
    for _ in range(n_phases):
        bounds.append((-10.0, 10.0))   # z
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))    # r_max (normalized)
    for _ in range(n_phases):
        bounds.append((0.0, 1.2))      # lambda (normalized)

    seed = 42
    pop = 30
    init_pop = np.tile(theta0n, (pop, 1)) * np.random.default_rng(seed).uniform(0.9, 1.1, (pop, len(theta0n)))

    res_de = differential_evolution(
        sse_loss,
        bounds=bounds,
        args=(t_norm, y_norm, term_func, n_phases),
        maxiter=1200,
        popsize=pop,
        init=init_pop,
        seed=seed,
        tol=1e-6,
        polish=True
    )

    res = minimize(
        sse_loss,
        x0=res_de.x,
        args=(t_norm, y_norm, term_func, n_phases),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-10
    )

    theta_n = res.x
    theta = np.zeros_like(theta_n)
    theta[0] = theta_n[0] * y_scale
    theta[1] = theta_n[1] * y_scale
    theta[2:2+n_phases] = theta_n[2:2+n_phases]
    theta[2+n_phases:2+2*n_phases] = theta_n[2+n_phases:2+2*n_phases] * (y_scale / t_scale)
    theta[2+2*n_phases:2+3*n_phases] = theta_n[2+2*n_phases:2+3*n_phases] * t_scale

    yhat = polyauxic_model(t, theta, term_func, n_phases)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    N = len(y)
    k = len(theta)
    aic = N * np.log(max(sse / N, 1e-12)) + 2*k
    bic = N * np.log(max(sse / N, 1e-12)) + k*np.log(N)
    aicc = aic + (2*k*(k+1))/(N-k-1) if (N-k-1) > 0 else np.inf

    return {
        "n_phases": n_phases,
        "theta": theta,
        "y_pred": yhat,
        "metrics": {"SSE": sse, "R2": r2, "AIC": aic, "AICc": aicc, "BIC": bic}
    }

def robust_prefit_for_outliers(t, y, term_func, n_phases):
    """
    Robust pre-fit with Soft L1 (Charbonnier-like) used ONLY for outlier detection.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n_params = 2 + 3*n_phases
    if len(t) <= n_params + 2:
        return None

    t_scale = float(np.max(t)) if float(np.max(t)) > 0 else 1.0
    y_scale = float(np.max(y)) if float(np.max(y)) > 0 else 1.0
    t_norm = t / t_scale
    y_norm = y / y_scale

    theta0 = smart_initial_guess(t, y, n_phases)
    theta0n = np.zeros_like(theta0)
    theta0n[0] = theta0[0] / y_scale
    theta0n[1] = theta0[1] / y_scale
    theta0n[2:2+n_phases] = 0.0
    theta0n[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases] / (y_scale / t_scale)
    theta0n[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases] / t_scale

    bounds = []
    bounds.append((0.0, 2.0))
    bounds.append((0.0, 2.0))
    for _ in range(n_phases):
        bounds.append((-10.0, 10.0))
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))
    for _ in range(n_phases):
        bounds.append((0.0, 1.2))

    seed = 42
    pop = 30
    init_pop = np.tile(theta0n, (pop, 1)) * np.random.default_rng(seed).uniform(0.9, 1.1, (pop, len(theta0n)))

    res_de = differential_evolution(
        soft_l1_loss,
        bounds=bounds,
        args=(t_norm, y_norm, term_func, n_phases),
        maxiter=900,
        popsize=pop,
        init=init_pop,
        seed=seed,
        tol=1e-6,
        polish=True
    )

    res = minimize(
        soft_l1_loss,
        x0=res_de.x,
        args=(t_norm, y_norm, term_func, n_phases),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-10
    )

    theta_n = res.x
    theta = np.zeros_like(theta_n)
    theta[0] = theta_n[0] * y_scale
    theta[1] = theta_n[1] * y_scale
    theta[2:2+n_phases] = theta_n[2:2+n_phases]
    theta[2+n_phases:2+2*n_phases] = theta_n[2+n_phases:2+2*n_phases] * (y_scale / t_scale)
    theta[2+2*n_phases:2+3*n_phases] = theta_n[2+2*n_phases:2+3*n_phases] * t_scale

    yhat = polyauxic_model(t, theta, term_func, n_phases)
    return {"theta": theta, "y_pred": yhat}

# =========================
# Data ingestion
# =========================

def load_pairs_dataframe(df):
    """
    Expected format:
    column pairs: (t1, y1), (t2, y2), ...
    header required. Up to any number of pairs (proof-of-concept).
    """
    df = df.reset_index(drop=True)
    ncols = df.shape[1]
    nrep = ncols // 2

    reps = []
    all_t = []
    all_y = []

    for i in range(nrep):
        t_raw = pd.to_numeric(df.iloc[:, 2*i], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(df.iloc[:, 2*i + 1], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(t_raw) & ~np.isnan(y_raw)
        t = t_raw[mask]
        y = y_raw[mask]
        if len(t) == 0:
            continue
        reps.append({"t": t, "y": y, "name": f"Replicate {len(reps)+1}"})
        all_t.append(t)
        all_y.append(y)

    if len(all_t) == 0:
        return np.array([]), np.array([]), []

    t_flat = np.concatenate(all_t)
    y_flat = np.concatenate(all_y)
    idx = np.argsort(t_flat)
    return t_flat[idx], y_flat[idx], reps

# =========================
# Plotting
# =========================

def plot_summary(reps, best_b, best_g):
    fig, ax = plt.subplots(figsize=(8, 5))

    # raw points
    for i, rep in enumerate(reps):
        ax.scatter(rep["t"], rep["y"], facecolors="white", edgecolors="black", s=18, alpha=0.7, label="Data" if i == 0 else None)

    t_max = max([float(r["t"].max()) for r in reps]) if reps else 1.0
    t_s = np.linspace(0.0, t_max, 300)

    # overlay best fits
    if best_g is not None:
        y_s = polyauxic_model(t_s, best_g["theta"], gompertz_term_eq32, best_g["n_phases"])
        ax.plot(t_s, y_s, linewidth=2.0, label=f"Gompertz: {best_g['n_phases']} phases (AICc {best_g['metrics']['AICc']:.1f})")

    if best_b is not None:
        y_s = polyauxic_model(t_s, best_b["theta"], boltzmann_term_eq31, best_b["n_phases"])
        ax.plot(t_s, y_s, linewidth=2.0, label=f"Boltzmann: {best_b['n_phases']} phases (AICc {best_b['metrics']['AICc']:.1f})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Response")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

def svg_download(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    st.download_button("Download SVG", data=buf.getvalue(), file_name=filename, mime="image/svg+xml")

# =========================
# Streamlit App
# =========================

def main():
    st.set_page_config(page_title="Polyauxic Modeling Platform (PoC)", layout="wide")
    st.title("Polyauxic Modeling Platform (Proof of Concept)")

    st.info(
        "This minimal Streamlit application demonstrates the core workflow described in the manuscript:\n"
        "- Data ingestion from CSV/XLSX using paired columns (t1,y1,t2,y2,...)\n"
        "- Polyauxic regression using Eq. 31 (Boltzmann) and Eq. 32 (Gompertz) with Softmax weights (Eq. 33)\n"
        "- Two-stage optimization (Differential Evolution → L-BFGS-B)\n"
        "- Outlier detection using ROUT (FDR), with Q fixed at 1.0%\n"
        "- Model selection via AIC/AICc/BIC and the first local minimum rule\n"
    )

    with st.expander("Instructions & File Format", expanded=False):
        st.markdown(
            "**File types:** `.csv` or `.xlsx`  \n"
            "**Required layout:** paired columns: `t1, y1, t2, y2, ...` (time followed by response).  \n"
            "**Header:** first row must contain column names.  \n"
            "**Replicates:** each (t,y) pair is treated as a replicate.  \n"
            "**Decimals:** both dot and comma are accepted (comma will be parsed if your CSV uses it via Excel export; if needed, save as standard CSV with dots)."
        )

    st.sidebar.header("Settings")
    max_phases = st.sidebar.number_input("Max phases to test", min_value=1, max_value=10, value=5)
    use_outliers = st.sidebar.checkbox("Detect & remove outliers (ROUT, Q=1.0%)", value=True)

    uploaded = st.sidebar.file_uploader("Upload CSV/XLSX (pairs: t1,y1,t2,y2,...)", type=["csv", "xlsx"])

    if not uploaded:
        st.warning("Upload a file to start.")
        footer()
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        t, y, reps = load_pairs_dataframe(df)
        if len(reps) == 0:
            st.error("No valid (t,y) column pairs were detected.")
            footer()
            return

        st.success(f"Loaded {len(reps)} replicate(s). Total points: {len(y)}")

        if st.button("RUN ANALYSIS"):
            st.divider()

            best_b = None
            best_g = None

            for model_name, term in [("Gompertz", gompertz_term_eq32), ("Boltzmann", boltzmann_term_eq31)]:
                st.subheader(f"{model_name} (Eq. {'32' if model_name=='Gompertz' else '31'})")

                results = []
                for n in range(1, int(max_phases) + 1):
                    with st.spinner(f"Fitting {n} phase(s)..."):
                        t_fit = t.copy()
                        y_fit = y.copy()

                        if use_outliers:
                            pre = robust_prefit_for_outliers(t_fit, y_fit, term, n)
                            if pre is not None:
                                mask = detect_outliers_rout_fdr(y_fit, pre["y_pred"], Q=1.0)  # fixed
                                # require enough points after removal
                                if np.any(mask) and (len(y_fit[~mask]) > (2 + 3*n + 5)):
                                    t_fit = t_fit[~mask]
                                    y_fit = y_fit[~mask]

                        res = fit_model(t_fit, y_fit, term, n)
                        if res is not None:
                            results.append(res)

                if len(results) == 0:
                    st.warning("Insufficient data to fit any phase count.")
                    continue

                # choose IC
                N = len(y)
                k_list = [2 + 3*r["n_phases"] for r in results]
                ic = choose_information_criterion(N, max(k_list))
                ic_vals = [r["metrics"][ic] for r in results]

                idx = first_local_min_index(ic_vals)
                best = results[idx]

                # display table
                table = pd.DataFrame([{
                    "Phases": r["n_phases"],
                    "R2": r["metrics"]["R2"],
                    "SSE": r["metrics"]["SSE"],
                    "AIC": r["metrics"]["AIC"],
                    "AICc": r["metrics"]["AICc"],
                    "BIC": r["metrics"]["BIC"],
                    f"{ic} (used)": r["metrics"][ic]
                } for r in results])

                st.dataframe(table.style.format({
                    "R2": "{:.4f}", "SSE": "{:.4g}", "AIC": "{:.2f}", "AICc": "{:.2f}", "BIC": "{:.2f}", f"{ic} (used)": "{:.2f}"
                }), hide_index=True)

                st.success(f"Best suggested model: {best['n_phases']} phase(s) (first local minimum of {ic}).")

                # plot best fit
                fig, ax = plt.subplots(figsize=(8, 4.5))
                for i, rep in enumerate(reps):
                    ax.scatter(rep["t"], rep["y"], facecolors="white", edgecolors="black", s=18, alpha=0.7, label="Data" if i == 0 else None)

                t_s = np.linspace(0.0, float(t.max()), 300)
                y_s = polyauxic_model(t_s, best["theta"], term, best["n_phases"])
                ax.plot(t_s, y_s, linewidth=2.2, label="Global fit")
                ax.set_xlabel("Time")
                ax.set_ylabel("Response")
                ax.grid(True, linestyle=":", alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                svg_download(fig, f"{model_name.lower()}_best_fit.svg")

                if model_name == "Boltzmann":
                    best_b = best
                else:
                    best_g = best

            st.subheader("Overall summary")
            plot_summary(reps, best_b, best_g)

    except Exception as e:
        st.error(f"Error processing data: {e}")

    footer()

# =========================
# Footer (kept as requested)
# =========================

def footer():
    profile_pic_url = "https://github.com/gusmock.png"
    st.markdown("---")

    footer_html = f"""
    <style>
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
        .profile-section {{
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            max-width: 800px;
        }}
        @media (max-width: 600px) {{
            .profile-section {{ flex-direction: column; }}
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

if __name__ == "__main__":
    main()

