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
# 2. FUNÇÕES DE SUPORTE E OTIMIZAÇÃO (FOCADAS EM R2)
# ==============================================================================

def sse_loss(theta, t, y, model_func, n_phases):
    """
    Soma dos Erros Quadrados (SSE).
    Minimizar isso garante o maior R² matematicamente possível.
    """
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    # Penalidade suave para valores negativos impossíveis biologicamente (opcional)
    if np.any(y_pred < -0.1 * np.max(y)):
        return 1e12
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Hessiana Numérica para cálculo de erro."""
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
    """Detecta outliers visualmente (ROUT method simplificado)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def smart_initial_guess(t, y, n_phases):
    """Heurística para encontrar onde as fases começam."""
    dy = np.gradient(y, t)
    # Suaviza a derivada para evitar ruído
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
    
    # Procura picos na derivada (máxima taxa de crescimento)
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth)*0.1, distance=min_dist)
    
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lam': t[p_idx], 'rmax': dy_smooth[p_idx]})
            
    # Preenche o resto se não achou picos suficientes
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lam': t.min() + t_span * (len(guesses)+1)/(n_phases+1),
            'rmax': (np.max(y)-np.min(y)) / (t_span/n_phases) # Estimativa linear grosseira
        })
    guesses.sort(key=lambda x: x['lam'])
    
    # Monta o vetor theta inicial
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y)
    theta_guess[1] = np.max(y)
    theta_guess[2:2+n_phases] = 0.0 # Logits iguais (pesos iguais)
    
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['rmax']
        theta_guess[2+2*n_phases+i] = guesses[i]['lam']
        
    return theta_guess

# ==============================================================================
# 3. FIT ENGINE (SSE OPTIMIZATION)
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases):
    
    # 1. Normalização (Essencial para o solver)
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    # 2. Inicialização Inteligente
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    
    # Converter para espaço normalizado
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    # Cria população inicial robusta
    pop_size = 50 # AUMENTADO para garantir varredura global
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    # Variação de +/- 20% em torno do smart guess
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    # Bounds mais permissivos para evitar travar no limite
    bounds = []
    bounds.append((-0.2, 1.5)) # yi (permite leve negativo para ajuste)
    bounds.append((0.0, 2.0))  # yf
    for _ in range(n_phases): bounds.append((-10, 10))   # z (softmax logits)
    for _ in range(n_phases): bounds.append((0.0, 500.0)) # rmax (pode ser muito alto se normalizado)
    for _ in range(n_phases): bounds.append((-0.1, 1.2))  # lam (permite latência levemente negativa ou pós fim)

    # 3. Otimização Global (Focada em SSE para R2 máximo)
    # differential_evolution é ótimo para fugir de mínimos locais
    res_de = differential_evolution(
        sse_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=None, # Seed aleatória para tentar caminhos novos
        polish=True,
        tol=1e-5
    )
    
    # 4. Refinamento Local Final
    res_opt = minimize(
        sse_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-9
    )
    
    theta_norm = res_opt.x
    
    # 5. Desnormalização
    theta_real = np.zeros_like(theta_norm)
    theta_real[0] = theta_norm[0] * y_scale
    theta_real[1] = theta_norm[1] * y_scale
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_scale
    
    # 6. Hessiana e Erros
    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        
        # Inversão segura
        cov_norm = sigma2 * np.linalg.pinv(H_norm) # pinv é mais estável que inv
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
    
    # Detecta outliers "post-hoc" apenas para visualização
    outliers = detect_outliers(y_data, y_pred)
    
    sse = np.sum((y_data - y_pred)**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - sse/sst
    
    n = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12 # Evita log(0)
    
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
        # Plota todos os pontos (inclusive outliers)
        ax.scatter(t, y, color='black', alpha=0.3, s=30, label='Dados Totais')
        
        # Marca os outliers
        if np.any(mask):
            ax.scatter(t[mask], y[mask], color='red', marker='x', s=60, linewidth=2, label='Outliers (Estatístico)')
        
        t_smooth = np.linspace(t.min(), t.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label='Ajuste SSE (Max R²)')
        
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
    st.set_page_config(layout="wide", page_title="Polyauxic Max-R2")
    st.title("Modelagem Poliauxica (Maximização de R²)")
    st.markdown("""
    Esta versão utiliza **Soma dos Erros Quadrados (SSE)** para o ajuste, garantindo a curva mais próxima possível dos pontos.
    Os outliers são identificados estatisticamente *após* o ajuste.
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
    
    if st.button("INICIAR AJUSTE"):
        st.divider()
        tab_g, tab_b = st.tabs(["Gompertz", "Boltzmann"])
        
        # --- LOOP GOMPERTZ ---
        with tab_g:
            prev_r2 = -np.inf
            n = 1
            while n <= 6:
                with st.expander(f"Gompertz: Tentativa com {n} Fase(s)", expanded=True):
                    with st.spinner(f"Maximizando R² para {n} fases..."):
                        res = fit_model_auto(t, y, gompertz_term_eq32, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, gompertz_term_eq32, "tab:blue")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada: Ganho de R² ({delta_r2:.4f}) < 1%. Sugere-se {n-1} fases.")
                                break
                            else:
                                st.success(f"✅ R² melhorou {delta_r2:.4f}. Continuando...")
                        
                        prev_r2 = curr_r2
                        n += 1

        # --- LOOP BOLTZMANN ---
        with tab_b:
            prev_r2 = -np.inf
            n = 1
            while n <= 6:
                with st.expander(f"Boltzmann: Tentativa com {n} Fase(s)", expanded=True):
                    with st.spinner(f"Maximizando R² para {n} fases..."):
                        res = fit_model_auto(t, y, boltzmann_term_eq31, n)
                        curr_r2 = res['metrics']['R2']
                        delta_r2 = curr_r2 - prev_r2
                        
                        display_analysis(res, n, t, y, boltzmann_term_eq31, "tab:orange")
                        
                        if n > 1:
                            if delta_r2 < 0.01:
                                st.warning(f"⏹️ Parada: Ganho de R² ({delta_r2:.4f}) < 1%. Sugere-se {n-1} fases.")
                                break
                            else:
                                st.success(f"✅ R² melhorou {delta_r2:.4f}. Continuando...")
                        
                        prev_r2 = curr_r2
                        n += 1

if __name__ == "__main__":
    main()
