import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks

# ==============================================================================
# 1. MODELOS MATEMÁTICOS (EQ. 31 e 32)
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    
    z = theta[2 : 2+n_phases]
    r_max = theta[2+n_phases : 2+2*n_phases]
    lambda_ = theta[2+2*n_phases : 2+3*n_phases]

    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])

    return y_i + (y_f - y_i) * sum_phases

# ==============================================================================
# 2. FUNÇÃO DE PERDA TUNÁVEL (LORENTZIANA RELAXADA)
# ==============================================================================

def lorentzian_loss(theta, t, y, model_func, n_phases, robustness_factor):
    """
    Perda Lorentziana com Fator de Escala Ajustável.
    
    robustness_factor: Multiplicador do MAD.
       1.0 = Padrão Estatístico (Rigoroso com outliers)
       >1.0 = Mais relaxado (Tende a SSE, aceita mais desvios antes de cortar)
    """
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    residuals = y - y_pred
    
    # 1.4826 é o fator de consistência para distribuição normal
    # Multiplicamos pelo fator do usuário para alargar a "bacia" de atração
    mad = np.median(np.abs(residuals - np.median(residuals)))
    
    # Se MAD for muito zero (ajuste perfeito ou inicial), define escala mínima
    if mad < 1e-9: mad = 1e-9
        
    scale = robustness_factor * 1.4826 * mad
    
    # Perda logarítmica: log(1 + (erro/scale)^2)
    loss = np.sum(np.log(1.0 + (residuals / scale)**2))
    return loss

def sse_loss(theta, t, y, model_func, n_phases):
    """Usada apenas para cálculo de erro padrão e R2 final"""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
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
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
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
    
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y)
    theta_guess[1] = np.max(y)
    theta_guess[2:2+n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['r_max']
        theta_guess[2+2*n_phases+i] = guesses[i]['lambda']
    return theta_guess

# ==============================================================================
# 3. MOTOR DE AJUSTE (COM FATOR DE ROBUSTEZ)
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases, robustness_factor):
    
    # 1. Normalização
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    # 2. Inicialização
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    pop_size = 50 
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    bounds = []
    bounds.append((-0.2, 1.5)) 
    bounds.append((0.0, 2.0))
    for _ in range(n_phases): bounds.append((-10, 10))
    for _ in range(n_phases): bounds.append((0.0, 500.0))
    for _ in range(n_phases): bounds.append((-0.1, 1.2))

    # Argumentos passados para a loss function (incluindo o fator)
    loss_args = (t_norm, y_norm, model_func, n_phases, robustness_factor)

    # 3. Otimização Global (Lorentziana Tunável)
    res_de = differential_evolution(
        lorentzian_loss,
        bounds,
        args=loss_args,
        maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
        seed=None, polish=True, tol=1e-5
    )
    
    # 4. Refinamento Local
    res_opt = minimize(
        lorentzian_loss,
        res_de.x,
        args=loss_args,
        method='L-BFGS-B', bounds=bounds, tol=1e-10
    )
    
    theta_norm = res_opt.x
    
    # 5. Desnormalização e Erros
    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)

    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    scale_r = y_scale / t_scale
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * scale_r
    scale_l = t_scale
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * scale_l
    
    try:
        # Hessiana calculada via SSE (convenção para erro padrão)
        # Note que o ajuste foi robusto, mas o erro padrão assume normalidade local no ótimo
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2:2+n_phases] = se_norm[2:2+n_phases]
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * scale_r
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * scale_l
    except:
        se_real = np.full_like(theta_real, np.nan)

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
        "n_phases": n_phases,
        "theta": theta_real,
        "se": se_real,
        "metrics": {"R2": r2, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. INTERFACE E VISUALIZAÇÃO
# ==============================================================================

def display_single_fit(res, t, y, model_func, color_main):
    n = res['n_phases']
    theta = res['theta']
    se = res['se']
    mask = res['outliers']
    
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
            y_ind = model_func(t_smooth, y_i, y_f, ph['p'], ph['r_max'], ph['lambda'])
            y_vis = y_i + (y_f - y_i) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.6, label=f'Fase {i+1}')
        
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.3)
        st.pyplot(fig)
        
    with c_data:
        df_glob = pd.DataFrame({
            "Param": ["y_i", "y_f"], "Valor": [y_i, y_f], "SE": [y_i_se, y_f_se]
        })
        st.dataframe(df_glob.style.format({"Valor": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "F": i+1,
                "p": ph['p'],
                "r_max": ph['r_max'], "SE r_max": ph['r_max_se'],
                "λ": ph['lambda'], "SE λ": ph['lambda_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", "r_max": "{:.4f}", "SE r_max": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Relaxed")
    st.title("Modelagem Poliauxica Robusta (Lorentziana Ajustável)")
    st.markdown("""
    **Problema:** A Lorentziana padrão pode ser agressiva demais e ignorar picos reais.
    **Solução:** Ajuste o **Fator de Relaxamento**. 
    * **1.0**: Padrão (Rigoroso).
    * **>1.0**: Mais permissivo (Aumenta o R² em casos de ajustes difíceis).
    """)
    
    # SLIDER DE ROBUSTEZ
    robustness_factor = st.sidebar.slider(
        "Fator de Relaxamento (Robustez)", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="Aumente este valor se o ajuste estiver 'achatado' ou com R² muito baixo. Valores maiores aproximam o método de Mínimos Quadrados."
    )
    
    file = st.sidebar.file_uploader("Arquivo CSV/XLSX", type=["csv", "xlsx"])
    max_phases = st.sidebar.number_input("Máximo de Fases", 1, 6, 5)
    
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
    
    if st.button("EXECUTAR ANÁLISE"):
        st.divider()
        tab_g, tab_b = st.tabs(["Gompertz (Eq. 32)", "Boltzmann (Eq. 31)"])
        
        def run_model_loop(model_name, model_func, color):
            results_list = []
            
            for n in range(1, max_phases + 1):
                with st.expander(f"{model_name}: {n} Fase(s)", expanded=False):
                    # Passando o fator de robustez
                    res = fit_model_auto(t, y, model_func, n, robustness_factor)
                    if res is None:
                        st.warning("Dados insuficientes.")
                        continue
                    display_single_fit(res, t, y, model_func, color)
                    results_list.append(res)
            
            st.markdown("### Seleção de Modelo")
            summary_data = []
            best_aicc = np.inf
            
            for i, res in enumerate(results_list):
                m = res['metrics']
                summary_data.append({
                    "Fases": res['n_phases'],
                    "R²": m['R2'],
                    "SSE": m['SSE'],
                    "AIC": m['AIC'],
                    "AICc": m['AICc'],
                    "BIC": m['BIC']
                })
                if m['AICc'] < best_aicc: best_aicc = m['AICc']
            
            def highlight_best(row):
                return ['background-color: #d4edda; font-weight: bold'] * len(row) if row['AICc'] == best_aicc else [''] * len(row)
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary.style.apply(highlight_best, axis=1).format({
                "R²": "{:.4f}", "SSE": "{:.4f}", 
                "AIC": "{:.4f}", "AICc": "{:.4f}", "BIC": "{:.4f}"
            }), hide_index=True)

        with tab_g:
            run_model_loop("Gompertz", gompertz_term_eq32, "tab:blue")
            
        with tab_b:
            run_model_loop("Boltzmann", boltzmann_term_eq31, "tab:orange")

if __name__ == "__main__":
    main()
