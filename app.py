import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks

# ==============================================================================
# 1. MODELOS MATEMÁTICOS (EQ. 31 e 32)
# ==============================================================================

def boltzmann_term_eq31(t, yi, yf, p_j, rmax_j, lam_j):
    """Modelo Boltzmann (Eq. 31) - Termo da fase j"""
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = 4.0 * rmax_j * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, yi, yf, p_j, rmax_j, lam_j):
    """Modelo Gompertz (Eq. 32) - Termo da fase j"""
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = rmax_j * np.e * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """Modelo Global (Soma das fases)"""
    t = np.asarray(t, dtype=float)
    yi = theta[0]
    yf = theta[1]
    
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * sum_phases

# ==============================================================================
# 2. FUNÇÕES DE SUPORTE E OTIMIZAÇÃO
# ==============================================================================

def lorentzian_loss(theta, t, y, model_func, n_phases):
    """Perda robusta (M-estimator)"""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    residuals = y - y_pred
    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = 1.4826 * mad
    if scale < 1e-6: scale = 1.0
    loss = np.sum(np.log(1.0 + (residuals / scale)**2))
    return loss

def sse_loss(theta, t, y, model_func, n_phases):
    """Perda SSE para Hessiana"""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Hessiana Numérica"""
    k = len(theta)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i = np.zeros(k); e_i[i] = epsilon
            e_j = np.zeros(k); e_j[j] = epsilon
            f_pp = func(theta + e_i + e_j, *args)
            f_pm = func(theta + e_i - e_j, *args)
            f_mp = func(theta - e_i + e_j, *args)
            f_mm = func(theta - e_i - e_j, *args)
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
    return hess

def detect_outliers(y_true, y_pred):
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def smart_initial_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    dy_smooth = np.convolve(dy, np.ones(3)/3, mode='same')
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth)*0.02, distance=min_dist)
    
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lam': t[p_idx], 'rmax': dy_smooth[p_idx]})
            
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lam': t.min() + t_span * (len(guesses)+1)/(n_phases+1),
            'rmax': np.mean(dy_smooth[dy_smooth>0]) if np.any(dy_smooth>0) else 1.0
        })
    guesses.sort(key=lambda x: x['lam'])
    
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y)
    theta_guess[1] = np.max(y)
    theta_guess[2:2+n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['rmax']
        theta_guess[2+2*n_phases+i] = guesses[i]['lam']
    return theta_guess

# ==============================================================================
# 3. FIT ENGINE (MAX ITERATIONS INDETERMINADAS VIA HARD LIMIT ALTO)
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases):
    # Configuração de "Iterações Indeterminadas" para o otimizador
    # Usamos um limite muito alto e confiamos na tolerância de convergência interna
    INTERNAL_MAX_ITER = 5000 
    
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    pop_size = 20
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.85, 1.15, init_pop.shape)

    bounds = []
    bounds.append((0.0, 1.3))
    bounds.append((0.0, 1.5))
    for _ in range(n_phases): bounds.append((-5, 5))
    for _ in range(n_phases): bounds.append((0, 50.0))
    for _ in range(n_phases): bounds.append((0, 1.1))

    # Otimização Global (Differential Evolution)
    res_de = differential_evolution(
        lorentzian_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=INTERNAL_MAX_ITER, # "Indeterminado" / Alto
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=42,
        polish=False,
        tol=1e-4 # Tolerância fina para convergência matemática
    )
    
    # Refinamento Local
    res_opt = minimize(
        lorentzian_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    theta_norm = res_opt.x
    
    # Desnormalização
    theta_real = np.zeros_like(theta_norm)
    theta_real[0] = theta_norm[0] * y_scale
    theta_real[1] = theta_norm[1] * y_scale
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_scale
    
    # Hessiana e Erros
    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        sigma2 = sse_val_norm / (len(y_norm) - len(theta_norm)) if len(y_norm) > len(theta_norm) else 1e-9
        cov_norm = sigma2 * np.linalg.inv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        
        se_real = np.zeros_like(se_norm)
        se_real[0] = se_norm[0] * y_scale
        se_real[1] = se_norm[1] * y_scale
        se_real[2:2+n_phases] = se_norm[2:2+n_phases]
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * t_scale
    except:
        se_real = np.full_like(theta_real, np.nan)

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)
    
    sse = np.sum((y_data - y_pred)**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - sse/sst
    
    n = len(y_data)
    k = len(theta_real)
    if sse <= 0: sse = 1e-9
    aic = n * np.log(sse/n) + 2*k
    bic = n * np.log(sse/n) + k * np.log(n)
    aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf
    
    return {
        "theta": theta_real,
        "se": se_real,
        "metrics": {"R2": r2, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. INTERFACE E VISUALIZAÇÃO
# ==============================================================================

def display_analysis(res, n, t, y, model_func, color_main):
    theta = res['theta']
    se = res['se']
    mask = res['outliers']
    
    yi, yf = theta[0], theta[1]
    yi_se, yf_se = se[0], se[1]
    
    z = theta[2:2+n]
    rmax = theta[2+n:2+2*n]
    rmax_se = se[2+n:2+2*n]
    lam = theta[2+2*n:2+3*n]
    lam_se = se[2+2*n:2+3*n]
    
    p = np.exp(z - np.max(z))
    p /= np.sum(p)
    
    phases = []
    for i in range(n):
        phases.append({
            "p": p[i],
            "rmax": rmax[i], "rmax_se": rmax_se[i],
            "lam": lam[i], "lam_se": lam_se[i]
        })
    phases.sort(key=lambda x: x['lam'])
    
    c_plot, c_data = st.columns([1.5, 1])
    
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(t[~mask], y[~mask], color='black', alpha=0.6, s=40, label='Dados Válidos')
        if np.any(mask):
            ax.scatter(t[mask], y[mask], color='red', marker='x', s=60, linewidth=2, label='Outliers')
        
        t_smooth = np.linspace(t.min(), t.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label='Ajuste')
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            y_ind = model_func(t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
            y_vis = yi + (yf - yi) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.6, label=f'Fase {i+1}')
            
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Resposta")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.3)
        st.pyplot(fig)
        
    with c_data:
        m = res['metrics']
        df_met = pd.DataFrame({
            "Critério": ["R²", "AICc", "AIC", "BIC"],
            "Valor": [m['R2'], m['AICc'], m['AIC'], m['BIC']]
        })
        st.dataframe(df_met.style.format({"Valor": "{:.4f}"}), hide_index=True)
        
        df_glob = pd.DataFrame({"Param": ["y_i", "y_f"], "Valor": [yi, yf], "SE": [yi_se, yf_se]})
        st.dataframe(df_glob.style.format({"Valor": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "F": i+1,
                "p": ph['p'],
                "µ": ph['rmax'], "SE µ": ph['rmax_se'],
                "λ": ph['lam'], "SE λ": ph['lam_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", "µ": "{:.4f}", "SE µ": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Auto-Loop")
    st.title("Modelagem Poliauxica: Loop Automático")
    st.markdown("""
    **Critério de Parada:** O algoritmo adiciona fases sucessivamente até que o ganho de **R² seja menor que 1% (0.01)**.
    """)
    
    file = st.sidebar.file_uploader("Arquivo CSV/XLSX", type=["csv", "xlsx"])
    if not file: st.stop()
    
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        c1, c2 = st.columns(2)
        t_col = c1.selectbox("Tempo", df.columns)
        y_col = c2.selectbox("Resposta", df.columns, index=1)
        t = pd.to_numeric(df[t_col], errors='coerce').dropna().values
        y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
        idx = np.argsort(t); t=t[idx]; y=y[idx]
    except: st.error("Erro dados."); st.stop()
    
    if st.button("INICIAR AJUSTE AUTOMÁTICO"):
        st.divider()
        tab_g, tab_b = st.tabs(["Gompertz", "Boltzmann"])
        
        # --- LOOP GOMPERTZ ---
        with tab_g:
            prev_r2 = -np.inf
            n = 1
            while n <= 6: # Limite de segurança
                with st.expander(f"Gompertz: Tentativa com {n} Fase(s)", expanded=True):
                    with st.spinner(f"Otimizando {n} fases..."):
                        res = fit_model_auto(t, y, gompertz_term_eq32, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, gompertz_term_eq32, "tab:blue")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada Automática: Ganho de R² ({delta_r2:.4f}) foi inferior a 1%. O modelo recomendado é o anterior ({n-1} fases).")
                                break
                            else:
                                st.success(f"✅ Melhora significativa de R² ({delta_r2:.4f}). Continuando...")
                        
                        prev_r2 = curr_r2
                        n += 1

        # --- LOOP BOLTZMANN ---
        with tab_b:
            prev_r2 = -np.inf
            n = 1
            while n <= 6:
                with st.expander(f"Boltzmann: Tentativa com {n} Fase(s)", expanded=True):
                    with st.spinner(f"Otimizando {n} fases..."):
                        res = fit_model_auto(t, y, boltzmann_term_eq31, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, boltzmann_term_eq31, "tab:orange")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada Automática: Ganho de R² ({delta_r2:.4f}) foi inferior a 1%. O modelo recomendado é o anterior ({n-1} fases).")
                                break
                            else:
                                st.success(f"✅ Melhora significativa de R² ({delta_r2:.4f}). Continuando...")
                        
                        prev_r2 = curr_r2
                        n += 1

if __name__ == "__main__":
    main()
