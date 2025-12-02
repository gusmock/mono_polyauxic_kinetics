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
    """
    Termo da fase j para o modelo Boltzmann (Eq. 31).
    """
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Argumento da exponencial
    numerator = 4.0 * rmax_j * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    
    # Clip para estabilidade numérica
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, yi, yf, p_j, rmax_j, lam_j):
    """
    Termo da fase j para o modelo Gompertz (Eq. 32).
    """
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Argumento da exponencial interna
    numerator = rmax_j * np.e * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """
    Soma das fases ponderadas (Modelo Global).
    """
    t = np.asarray(t, dtype=float)
    yi = theta[0]
    yf = theta[1]
    
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax para p_j
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * sum_phases

# ==============================================================================
# 2. FUNÇÃO DE PERDA ROBUSTA E HEURÍSTICA AUTOMÁTICA
# ==============================================================================

def lorentzian_loss(theta, t, y, model_func, n_phases):
    """
    Função de perda robusta (Lorentziana) para ignorar outliers durante o ajuste.
    """
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    residuals = y - y_pred
    
    # Estimativa robusta da escala (MAD)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = 1.4826 * mad
    if scale < 1e-6: scale = 1.0 # Evita divisão por zero em ajustes perfeitos
    
    # Perda log-Lorentziana
    loss = np.sum(np.log(1.0 + (residuals / scale)**2))
    return loss

def smart_initial_guess(t, y, n_phases):
    """
    Inicialização Automática: Usa derivadas para encontrar picos e estimar parâmetros.
    """
    # Suavização e Derivada
    dy = np.gradient(y, t)
    # Filtro de média móvel simples
    dy_smooth = np.convolve(dy, np.ones(3)/3, mode='same')
    
    # Encontrar picos na derivada (pontos de inflexão = maior taxa)
    # Distância mínima relativa ao tamanho dos dados
    min_dist = max(1, len(t) // (n_phases * 3))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth)*0.05, distance=min_dist)
    
    guesses = []
    
    # Se encontrou picos reais, usa-os
    if len(peaks) > 0:
        # Pega os maiores picos
        sorted_indices = np.argsort(props['peak_heights'])[::-1] # Descendente
        best_peaks = peaks[sorted_indices][:n_phases]
        
        for p_idx in best_peaks:
            guesses.append({
                'lam': t[p_idx], 
                'rmax': dy_smooth[p_idx]
            })
            
    # Preenche com palpites genéricos se faltarem picos
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lam': t.min() + t_span * (len(guesses)+1)/(n_phases+1),
            'rmax': np.mean(dy_smooth[dy_smooth>0])
        })
    
    guesses.sort(key=lambda x: x['lam'])
    
    # Montar vetor theta inicial
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y)      # yi
    theta_guess[1] = np.max(y)      # yf
    theta_guess[2:2+n_phases] = 0.0 # z (probs iguais)
    
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['rmax']
        theta_guess[2+2*n_phases+i] = guesses[i]['lam']
        
    return theta_guess

# ==============================================================================
# 3. CÁLCULO DE ERROS (HESSIANA) E ESTATÍSTICAS
# ==============================================================================

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Calcula Hessiana numericamente para estimar Erro Padrão."""
    k = len(theta)
    hess = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            e_i = np.zeros(k); e_i[i] = epsilon
            e_j = np.zeros(k); e_j[j] = epsilon
            
            # Diferença finita centrada de segunda ordem
            f_pp = func(theta + e_i + e_j, *args)
            f_pm = func(theta + e_i - e_j, *args)
            f_mp = func(theta - e_i + e_j, *args)
            f_mm = func(theta - e_i - e_j, *args)
            
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            
    return hess

def sse_loss(theta, t, y, model_func, n_phases):
    """Perda quadrática simples (para cálculo da Hessiana/Variância)."""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    return np.sum((y - y_pred)**2)

def detect_outliers(y_true, y_pred):
    """Detecta outliers baseando-se no Desvio Absoluto da Mediana (MAD)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    
    # Sigma estimado robusto
    sigma_robust = 1.4826 * mad
    if sigma_robust < 1e-9: sigma_robust = 1e-9
    
    # Z-score robusto
    z_scores = np.abs(residuals - median_res) / sigma_robust
    
    # Threshold de 2.5 sigma (padrão ROUT moderado)
    is_outlier = z_scores > 2.5
    return is_outlier

# ==============================================================================
# 4. ROTINA DE AJUSTE
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases, max_iter=300):
    
    # 1. Normalização
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    # 2. Inicialização Inteligente (Smart Guess)
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    
    # Converter Smart Guess para Espaço Normalizado
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale # yi
    theta0_norm[1] = theta_smart[1] / y_scale # yf
    theta0_norm[2:2+n_phases] = 0.0           # z
    # rmax norm = rmax / (y/t)
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    # lam norm = lam / t
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    # População inicial centrada no Smart Guess
    pop_size = 20
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.85, 1.15, init_pop.shape) # +/- 15% de variação

    # Bounds normalizados
    bounds = []
    bounds.append((0.0, 1.3)) # yi
    bounds.append((0.0, 1.5)) # yf
    for _ in range(n_phases): bounds.append((-5, 5))   # z
    for _ in range(n_phases): bounds.append((0, 50.0)) # rmax
    for _ in range(n_phases): bounds.append((0, 1.1))  # lam

    # 3. Otimização Global (com Lorentzian Loss)
    res_de = differential_evolution(
        lorentzian_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=max_iter,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=42,
        polish=False
    )
    
    # 4. Refinamento Local
    res_opt = minimize(
        lorentzian_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    theta_norm = res_opt.x
    
    # 5. Desnormalização dos Parâmetros
    theta_real = np.zeros_like(theta_norm)
    theta_real[0] = theta_norm[0] * y_scale
    theta_real[1] = theta_norm[1] * y_scale
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_scale
    
    # 6. Cálculo de Erros Padrão (SE) via Hessiana
    # Usamos SSE Loss para a Hessiana pois a teoria assintótica assume normalidade
    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        # Variância do erro residual (MSE)
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        
        cov_norm = sigma2 * np.linalg.inv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        
        # Desnormalizar Erros
        se_real = np.zeros_like(se_norm)
        se_real[0] = se_norm[0] * y_scale
        se_real[1] = se_norm[1] * y_scale
        se_real[2:2+n_phases] = se_norm[2:2+n_phases]
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * t_scale
        
    except:
        se_real = np.full_like(theta_real, np.nan)

    # 7. Métricas Finais e Outliers
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outlier_mask = detect_outliers(y_data, y_pred)
    
    # Recalcula SSE apenas nos Inliers para AIC/BIC (Prática Comum em Robust Regression)
    # ou usa SSE total. Aqui usaremos SSE total para penalizar fit ruim, 
    # mas o fit foi guiado para ignorar o outlier.
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
        "outliers": outlier_mask,
        "y_pred": y_pred
    }

# ==============================================================================
# 5. INTERFACE
# ==============================================================================

def display_full_analysis(res, n, t, y, model_func, color_main):
    """Exibe gráfico com outliers em vermelho e tabelas de parâmetros."""
    
    theta = res['theta']
    se = res['se']
    mask = res['outliers']
    
    # Separação de parâmetros
    yi, yf = theta[0], theta[1]
    yi_se, yf_se = se[0], se[1]
    
    z = theta[2:2+n]
    rmax = theta[2+n:2+2*n]
    rmax_se = se[2+n:2+2*n]
    lam = theta[2+2*n:2+3*n]
    lam_se = se[2+2*n:2+3*n]
    
    # Softmax p
    p = np.exp(z - np.max(z))
    p /= np.sum(p)
    # Erro de p é complexo de propagar analiticamente, omitiremos ou deixaremos NaN
    
    # Organizar Fases
    phases = []
    for i in range(n):
        phases.append({
            "p": p[i],
            "rmax": rmax[i], "rmax_se": rmax_se[i],
            "lam": lam[i], "lam_se": lam_se[i]
        })
    phases.sort(key=lambda x: x['lam'])
    
    # --- LAYOUT ---
    c_plot, c_data = st.columns([1.5, 1])
    
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 1. Dados Válidos (Inliers)
        ax.scatter(t[~mask], y[~mask], color='black', alpha=0.6, s=40, label='Dados Válidos')
        
        # 2. Outliers (Vermelho X)
        if np.any(mask):
            ax.scatter(t[mask], y[mask], color='red', marker='x', s=60, linewidth=2, label='Outliers Detectados')
        
        # 3. Modelo Global
        t_smooth = np.linspace(t.min(), t.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label='Ajuste Robusto')
        
        # 4. Fases Individuais
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            y_ind = model_func(t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
            y_vis = yi + (yf - yi) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.7, label=f'Fase {i+1}')
            
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Resposta")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)
        st.pyplot(fig)
        
    with c_data:
        # Tabela Métricas
        st.markdown("**Critérios de Informação**")
        m = res['metrics']
        df_met = pd.DataFrame({
            "Critério": ["AICc", "AIC", "BIC", "R²", "SSE"],
            "Valor": [m['AICc'], m['AIC'], m['BIC'], m['R2'], m['SSE']]
        })
        st.dataframe(df_met.style.format({"Valor": "{:.4f}"}), hide_index=True)
        
        # Tabela Global
        st.markdown("**Parâmetros Globais**")
        df_glob = pd.DataFrame({
            "Param": ["y_i", "y_f"],
            "Valor": [yi, yf],
            "SE (+/-)": [yi_se, yf_se]
        })
        st.dataframe(df_glob.style.format("{:.4f}"), hide_index=True)
        
        # Tabela Fases
        st.markdown("**Fases (Ordenadas por Tempo)**")
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "Fase": i+1,
                "p": ph['p'],
                "µ_max": ph['rmax'], "SE µ": ph['rmax_se'],
                "λ": ph['lam'], "SE λ": ph['lam_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format("{:.4f}"), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Auto-Robust")
    st.title("Modelagem Poliauxica Automática e Robusta")
    st.markdown("""
    Esta ferramenta realiza o ajuste automático dos modelos **Gompertz (Eq. 32)** e **Boltzmann (Eq. 31)**.
    
    * **Inicialização:** Detecção automática de picos (sem input manual).
    * **Robustez:** Uso de perda Lorentziana para ignorar outliers no ajuste.
    * **Outliers:** Pontos estatisticamente discrepantes são marcados com **X vermelho**.
    """)
    
    st.sidebar.header("Carregar Dados")
    file = st.sidebar.file_uploader("Arquivo CSV ou Excel", type=["csv", "xlsx"])
    
    max_iter = st.sidebar.select_slider("Esforço Computacional", options=[100, 300, 500, 1000], value=300)
    
    if not file:
        st.info("Aguardando arquivo...")
        st.stop()
        
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        c1, c2 = st.columns(2)
        t_col = c1.selectbox("Coluna Tempo", df.columns)
        y_col = c2.selectbox("Coluna Resposta", df.columns, index=1)
        
        t = pd.to_numeric(df[t_col], errors='coerce').dropna().values
        y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
        
        # Ordenar e filtrar
        idx = np.argsort(t)
        t, y = t[idx], y[idx]
    except:
        st.error("Erro ao processar arquivo.")
        st.stop()
        
    if st.button("INICIAR ANÁLISE COMPLETA"):
        st.divider()
        
        tab_g, tab_b = st.tabs(["Resultados Gompertz", "Resultados Boltzmann"])
        
        # --- GOMPERTZ ---
        with tab_g:
            best_aic = np.inf
            for n in range(1, 6):
                with st.expander(f"Modelo Gompertz - {n} Fase(s)", expanded=(n==1)):
                    with st.spinner(f"Ajustando {n} fase(s)..."):
                        res = fit_model_auto(t, y, gompertz_term_eq32, n, max_iter)
                        display_full_analysis(res, n, t, y, gompertz_term_eq32, "tab:blue")
                        if res['metrics']['AICc'] < best_aic: best_aic = res['metrics']['AICc']
            st.success(f"Melhor AICc Gompertz: {best_aic:.4f}")

        # --- BOLTZMANN ---
        with tab_b:
            best_aic = np.inf
            for n in range(1, 6):
                with st.expander(f"Modelo Boltzmann - {n} Fase(s)", expanded=(n==1)):
                    with st.spinner(f"Ajustando {n} fase(s)..."):
                        res = fit_model_auto(t, y, boltzmann_term_eq31, n, max_iter)
                        display_full_analysis(res, n, t, y, boltzmann_term_eq31, "tab:orange")
                        if res['metrics']['AICc'] < best_aic: best_aic = res['metrics']['AICc']
            st.success(f"Melhor AICc Boltzmann: {best_aic:.4f}")

if __name__ == "__main__":
    main()
