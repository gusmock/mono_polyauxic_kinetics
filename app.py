import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks

# ==============================================================================
# 1. MODELOS MATEMÁTICOS (NOTAÇÃO EXATA DAS EQS. 31 E 32)
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Termo da fase j para o modelo Boltzmann (Eq. 31).
    Parâmetros:
      y_i      : Assíntota inicial
      y_f      : Assíntota final
      p_j      : Proporção da fase j
      r_max_j  : Taxa máxima da fase j
      lambda_j : Tempo de latência da fase j
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Eq. 31: expoente = [4 * r_max * (lambda - t)] / [(yf - yi) * p] + 2
    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    
    # Clip para estabilidade numérica
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Termo da fase j para o modelo Gompertz (Eq. 32).
    Parâmetros seguem a mesma notação da Eq. 31.
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Eq. 32: expoente interno = [r_max * e * (lambda - t)] / [(yf - yi) * p] + 1
    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """
    Modelo Global: Soma ponderada das fases sigmoidais.
    """
    t = np.asarray(t, dtype=float)
    
    # Mapeamento do vetor theta para a notação física
    y_i = theta[0]
    y_f = theta[1]
    
    # Vetores de parâmetros das fases
    z = theta[2 : 2+n_phases]                             # Parâmetros latentes para p_j
    r_max = theta[2+n_phases : 2+2*n_phases]              # r_max_j
    lambda_ = theta[2+2*n_phases : 2+3*n_phases]          # lambda_j

    # Softmax para garantir que a soma dos p_j seja 1
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # Somatório das fases
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])

    # Expressão final: y_i + (y_f - y_i) * Somatório
    return y_i + (y_f - y_i) * sum_phases

# ==============================================================================
# 2. FUNÇÕES DE PERDA E ESTATÍSTICAS
# ==============================================================================

def sse_loss(theta, t, y, model_func, n_phases):
    """
    Objective Function: Sum of Squared Errors (SSE).
    Minimizing SSE maximizes R² (assuming homoscedasticity).
    """
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    # Penalidade suave para evitar valores fisicamente impossíveis
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Cálculo numérico da Hessiana para estimativa de erros (Matriz de Fisher)."""
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
    """Deteção de outliers baseada no método ROUT (Robust Regression and Outlier Removal)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5 # Threshold conservador

def smart_initial_guess(t, y, n_phases):
    """Heurística baseada em derivadas para estimar r_max e lambda iniciais."""
    dy = np.gradient(y, t)
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
    
    # Encontra picos na derivada (pontos de inflexão)
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth)*0.1, distance=min_dist)
    
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lambda': t[p_idx], 'r_max': dy_smooth[p_idx]})
            
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lambda': t.min() + t_span * (len(guesses)+1)/(n_phases+1),
            'r_max': (np.max(y)-np.min(y)) / (t_span/n_phases)
        })
    guesses.sort(key=lambda x: x['lambda'])
    
    # Montagem do vetor theta inicial
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y) # y_i
    theta_guess[1] = np.max(y) # y_f
    theta_guess[2:2+n_phases] = 0.0 # z (pesos iguais)
    
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['r_max']
        theta_guess[2+2*n_phases+i] = guesses[i]['lambda']
        
    return theta_guess

# ==============================================================================
# 3. MOTOR DE AJUSTE (FIT ENGINE)
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases):
    
    # 1. Normalização
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    # 2. Inicialização
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale # y_i
    theta0_norm[1] = theta_smart[1] / y_scale # y_f
    theta0_norm[2:2+n_phases] = 0.0 # z
    
    # Normalização de r_max e lambda
    # r_max [unidade y/t] -> divide por (y_scale/t_scale)
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    # lambda [unidade t] -> divide por t_scale
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    # População do DE
    pop_size = 50 
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    # Limites (Bounds)
    bounds = []
    bounds.append((-0.2, 1.5)) # y_i
    bounds.append((0.0, 2.0))  # y_f
    for _ in range(n_phases): bounds.append((-10, 10))    # z
    for _ in range(n_phases): bounds.append((0.0, 500.0)) # r_max
    for _ in range(n_phases): bounds.append((-0.1, 1.2))  # lambda

    # 3. Otimização Global
    res_de = differential_evolution(
        sse_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=None,
        polish=True,
        tol=1e-5
    )
    
    # 4. Refinamento Local
    res_opt = minimize(
        sse_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-9
    )
    
    theta_norm = res_opt.x
    
    # 5. Desnormalização e Cálculo de Erros
    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)

    # Fatores de escala
    # y_i, y_f
    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y

    # z (adimensional)
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]

    # r_max
    scale_r = y_scale / t_scale
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * scale_r

    # lambda
    scale_l = t_scale
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * scale_l
    
    # Hessiana
    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        
        # Desnormaliza Erros
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2:2+n_phases] = se_norm[2:2+n_phases]
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * scale_r
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * scale_l
    except:
        se_real = np.full_like(theta_real, np.nan)

    # 6. Métricas Finais
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)
    
    sse = np.sum((y_data - y_pred)**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - sse/sst
    
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12
    
    aic = n_len * np.log(sse/n_len) + 2*k
    bic = n_len * np.log(sse/n_len) + k * np.log(n_len)
    aicc = aic + (2*k*(k+1))/(n_len-k-1) if (n_len-k-1)>0 else np.inf
    
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
    
    # Extração usando notação oficial
    y_i, y_f = theta[0], theta[1]
    y_i_se, y_f_se = se[0], se[1]
    
    z = theta[2:2+n]
    r_max = theta[2+n:2+2*n]
    r_max_se = se[2+n:2+2*n]
    lambda_ = theta[2+2*n:2+3*n]
    lambda_se = se[2+2*n:2+3*n]
    
    p = np.exp(z - np.max(z))
    p /= np.sum(p)
    
    phases = []
    for i in range(n):
        phases.append({
            "p": p[i],
            "r_max": r_max[i], "r_max_se": r_max_se[i],
            "lambda": lambda_[i], "lambda_se": lambda_se[i]
        })
    # Ordena fases pelo tempo de latência (cronologia)
    phases.sort(key=lambda x: x['lambda'])
    
    c_plot, c_data = st.columns([1.5, 1])
    
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(t, y, color='black', alpha=0.3, s=30, label='Dados')
        
        if np.any(mask):
            ax.scatter(t[mask], y[mask], color='red', marker='x', s=60, linewidth=2, label='Outliers')
        
        t_smooth = np.linspace(t.min(), t.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label='Ajuste Global')
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            # Plota fase individual
            y_ind = model_func(t_smooth, y_i, y_f, ph['p'], ph['r_max'], ph['lambda'])
            y_vis = y_i + (y_f - y_i) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.6, label=f'Fase {i+1}')
            
        ax.set_xlabel("Tempo (t)")
        ax.set_ylabel("Resposta (y)")
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
        
        # Tabela Global com Notação y_i, y_f
        df_glob = pd.DataFrame({
            "Parâmetro": ["y_i", "y_f"], 
            "Valor": [y_i, y_f], 
            "SE": [y_i_se, y_f_se]
        })
        st.dataframe(df_glob.style.format({"Valor": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        
        # Tabela Fases com Notação p, r_max, lambda
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "Fase": i+1,
                "p": ph['p'],
                "r_max": ph['r_max'], "SE r_max": ph['r_max_se'],
                "λ": ph['lambda'], "SE λ": ph['lambda_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", 
            "r_max": "{:.4f}", "SE r_max": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Notation Fix")
    st.title("Modelagem Poliauxica (Notação Oficial)")
    st.markdown("""
    Ajuste automático dos modelos **Gompertz (Eq. 32)** e **Boltzmann (Eq. 31)**.
    
    **Notação dos Parâmetros:**
    * $y_i$: Início do crescimento
    * $y_f$: Fim do crescimento
    * $p_j$: Proporção da fase $j$
    * $r_{max,j}$: Taxa máxima da fase $j$
    * $\lambda_j$: Tempo de latência da fase $j$
    """)
    
    file = st.sidebar.file_uploader("Arquivo CSV/XLSX", type=["csv", "xlsx"])
    if not file: st.stop()
    
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        c1, c2 = st.columns(2)
        t_col = c1.selectbox("Tempo (t)", df.columns)
        y_col = c2.selectbox("Resposta (y)", df.columns, index=1)
        t = pd.to_numeric(df[t_col], errors='coerce').dropna().values
        y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
        idx = np.argsort(t); t=t[idx]; y=y[idx]
    except: st.error("Erro dados."); st.stop()
    
    if st.button("INICIAR AJUSTE"):
        st.divider()
        tab_g, tab_b = st.tabs(["Gompertz (Eq. 32)", "Boltzmann (Eq. 31)"])
        
        # --- LOOP GOMPERTZ ---
        with tab_g:
            prev_r2 = -np.inf
            n = 1
            while n <= 6:
                with st.expander(f"Gompertz: {n} Fase(s)", expanded=True):
                    with st.spinner(f"Ajustando {n} fases..."):
                        res = fit_model_auto(t, y, gompertz_term_eq32, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, gompertz_term_eq32, "tab:blue")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada Automática (Ganho R² < 1%). Sugerido: {n-1} Fases.")
                                break
                            else:
                                st.success(f"✅ Melhora de R² ({delta_r2:.4f}). Continuando...")
                        prev_r2 = curr_r2
                        n += 1

        # --- LOOP BOLTZMANN ---
        with tab_b:
            prev_r2 = -np.inf
            n = 1
            while n <= 6:
                with st.expander(f"Boltzmann: {n} Fase(s)", expanded=True):
                    with st.spinner(f"Ajustando {n} fases..."):
                        res = fit_model_auto(t, y, boltzmann_term_eq31, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, boltzmann_term_eq31, "tab:orange")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada Automática (Ganho R² < 1%). Sugerido: {n-1} Fases.")
                                break
                            else:
                                st.success(f"✅ Melhora de R² ({delta_r2:.4f}). Continuando...")
                        prev_r2 = curr_r2
                        n += 1

if __name__ == "__main__":
    main()
