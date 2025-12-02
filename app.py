import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. DEFINIÇÃO MATEMÁTICA DOS MODELOS
# ==============================================================================

def boltzmann_phase_eq31(t, yi, yf, p, rmax, lam):
    """
    Fase individual do Modelo Boltzmann (Eq. 31).
    """
    delta_y = yf - yi
    # Evita divisão por zero
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p, 1e-12)

    exponent = 4.0 * rmax * (lam - t) / (delta_y * p_safe) + 2.0
    # Clip para evitar overflow/underflow numérico
    exponent = np.clip(exponent, -100.0, 100.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_phase_eq32(t, yi, yf, p, rmax, lam):
    """
    Fase individual do Modelo Gompertz (Eq. 32).
    """
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p, 1e-12)

    exponent = 1.0 + (rmax * np.e) * (lam - t) / (delta_y * p_safe)
    # Clip para evitar overflow na exponencial interna
    exponent = np.clip(exponent, -100.0, 100.0)

    inner = np.exp(exponent)
    return p_safe * np.exp(-inner)

def polyauxic_model(t, theta, phase_func, n_phases):
    """
    Modelo Poliauxico Genérico (Soma de Fases).
    theta: [yi, yf, z_1..z_n, rmax_1..rmax_n, lam_1..lam_n]
    """
    t = np.asarray(t, dtype=float)
    yi = theta[0]
    yf = theta[1]

    # Fatiamento dos parâmetros
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax para garantir que a soma dos pesos p seja 1
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # Soma das contribuições
    y_red = 0.0
    for j in range(n_phases):
        y_red += phase_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * y_red

# ==============================================================================
# 2. FUNÇÕES DE OTIMIZAÇÃO E ESTATÍSTICA
# ==============================================================================

def sse_loss(params, t, y, model_func, n_phases):
    """Soma dos Erros Quadrados (SSE) para otimização."""
    y_pred = polyauxic_model(t, params, model_func, n_phases)
    residuals = y - y_pred
    return np.sum(residuals**2)

def numerical_hessian(func, x, args=(), eps=1e-4):
    """Calcula a matriz Hessiana numericamente para estimar erros."""
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            fpp = func(x + ei + ej, *args)
            fpm = func(x + ei - ej, *args)
            fmp = func(x - ei + ej, *args)
            fmm = func(x - ei - ej, *args)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    return H

def fit_single_scenario(t_data, y_data, model_func, n_phases, de_maxiter=200, seed=42):
    """
    Realiza o ajuste para um modelo e um número de fases específico.
    Utiliza normalização interna para estabilidade.
    """
    rng = np.random.default_rng(seed)
    n_data = len(y_data)
    
    # 1. Normalização (Crucial para convergência)
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale

    # 2. Definição de Limites (Bounds) para dados normalizados
    # [yi, yf, z_1..z_n, rmax_1..rmax_n, lam_1..lam_n]
    bounds = []
    bounds.append((0, 1.2)) # yi
    bounds.append((0, 1.5)) # yf
    for _ in range(n_phases): bounds.append((-5, 5))   # z (softmax)
    for _ in range(n_phases): bounds.append((0, 50.0)) # rmax (norm)
    for _ in range(n_phases): bounds.append((0, 1.1))  # lam (norm)

    # 3. Otimização Global
    res_global = differential_evolution(
        sse_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
        maxiter=de_maxiter, popsize=15, strategy='best1bin',
        seed=int(rng.integers(0, 100000)), polish=False, tol=0.01
    )

    # 4. Refinamento Local
    res_local = minimize(
        sse_loss, res_global.x, args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B', bounds=bounds
    )
    theta_norm = res_local.x
    sse_norm = res_local.fun

    # 5. Cálculo dos Erros Padrão (na escala normalizada)
    sigma2 = sse_norm / (n_data - len(theta_norm)) if n_data > len(theta_norm) else 0
    se_norm = np.full_like(theta_norm, np.nan)
    
    try:
        H = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        cov_matrix = sigma2 * np.linalg.inv(H)
        se_norm = np.sqrt(np.abs(np.diag(cov_matrix)))
    except:
        pass # Se Hessiana falhar, erros ficam como NaN

    # 6. Desnormalização (Parâmetros e Erros)
    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(se_norm)

    # Fatores de escala
    # yi, yf: * y_scale
    theta_real[0:2] = theta_norm[0:2] * y_scale
    se_real[0:2]    = se_norm[0:2] * y_scale
    
    # z: adimensional, não muda
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    se_real[2:2+n_phases]    = se_norm[2:2+n_phases]
    
    # rmax: unidade y/t -> * y_scale / t_scale
    rmax_factor = y_scale / t_scale
    idx_r_start = 2 + n_phases
    idx_r_end   = 2 + 2*n_phases
    theta_real[idx_r_start:idx_r_end] = theta_norm[idx_r_start:idx_r_end] * rmax_factor
    se_real[idx_r_start:idx_r_end]    = se_norm[idx_r_start:idx_r_end] * rmax_factor
    
    # lam: unidade t -> * t_scale
    lam_factor = t_scale
    idx_l_start = idx_r_end
    idx_l_end   = 2 + 3*n_phases
    theta_real[idx_l_start:idx_l_end] = theta_norm[idx_l_start:idx_l_end] * lam_factor
    se_real[idx_l_start:idx_l_end]    = se_norm[idx_l_start:idx_l_end] * lam_factor

    # 7. Estatísticas Finais
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    residuals = y_data - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - (sse/sst) if sst > 1e-12 else 0.0

    # AICc
    n_params = len(theta_real)
    if sse > 0:
        log_l = -n_data/2 * (np.log(2 * np.pi * sse / n_data) + 1)
        aic = 2*n_params - 2*log_l
        aicc = aic + (2*n_params*(n_params+1))/(n_data - n_params - 1) if (n_data - n_params - 1) > 0 else np.inf
    else:
        aicc = np.inf

    return {
        "theta": theta_real,
        "se": se_real,
        "r2": r2,
        "sse": sse,
        "aicc": aicc,
        "y_pred": y_pred
    }

# ==============================================================================
# 3. INTERFACE E VISUALIZAÇÃO
# ==============================================================================

def display_result(res, t_data, y_data, model_func, n_phases, model_name):
    """Helper para exibir gráfico e tabela de um único resultado."""
    
    st.markdown(f"**R²:** {res['r2']:.4f} | **AICc:** {res['aicc']:.4f} | **SSE:** {res['sse']:.4f}")
    
    theta = res['theta']
    se = res['se']
    
    yi, yf = theta[0], theta[1]
    yi_se, yf_se = se[0], se[1]
    
    # Extração das fases
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]
    
    rmax_se = se[2+n_phases : 2+2*n_phases]
    lam_se = se[2+2*n_phases : 2+3*n_phases]
    
    # Softmax para calcular p
    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))
    
    # Ordenação Cronológica por Lambda
    phases_data = []
    for i in range(n_phases):
        phases_data.append({
            "p": p[i],
            "rmax": rmax[i], "rmax_se": rmax_se[i],
            "lam": lam[i],   "lam_se": lam_se[i]
        })
    
    phases_data.sort(key=lambda x: x["lam"])
    
    # Colunas para Gráfico e Tabela
    col_graph, col_table = st.columns([1.5, 1])
    
    with col_graph:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t_data, y_data, color='black', alpha=0.5, s=30, label='Dados')
        
        # Curva Suave
        t_smooth = np.linspace(t_data.min(), t_data.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n_phases)
        ax.plot(t_smooth, y_smooth, 'r-', linewidth=2, label='Ajuste Global')
        
        # Fases individuais (visualização)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(phases_data)))
        for i, ph in enumerate(phases_data):
            y_indiv = model_func(t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
            y_vis = yi + (yf - yi) * y_indiv
            ax.plot(t_smooth, y_vis, '--', color=colors[i], linewidth=1, label=f'Fase {i+1}')
            
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Resposta")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col_table:
        st.caption("Parâmetros Globais")
        df_glob = pd.DataFrame({
            "Param": ["y_i", "y_f"],
            "Valor": [yi, yf],
            "Erro": [yi_se, yf_se]
        })
        # CORREÇÃO: Formatação seletiva
        st.dataframe(df_glob.style.format({
            "Valor": "{:.4f}",
            "Erro": "{:.4f}"
        }), hide_index=True)
        
        st.caption("Parâmetros das Fases")
        rows = []
        for i, ph in enumerate(phases_data):
            rows.append({
                "Fase": i+1,
                "p": ph['p'],
                "µ_max": ph['rmax'], "Err µ": ph['rmax_se'],
                "λ": ph['lam'], "Err λ": ph['lam_se']
            })
        df_ph = pd.DataFrame(rows)
        # CORREÇÃO: Formatação seletiva
        st.dataframe(df_ph.style.format({
            "p": "{:.4f}",
            "µ_max": "{:.4f}",
            "Err µ": "{:.4f}",
            "λ": "{:.4f}",
            "Err λ": "{:.4f}"
        }), hide_index=True)

# ==============================================================================
# 4. LOOP PRINCIPAL
# ==============================================================================

def main():
    st.set_page_config(page_title="Poliauxico Multi-Fase", layout="wide")
    st.title("Análise Poliauxica Completa (1 a 5 Fases)")
    st.markdown("Ajuste simultâneo de **Gompertz** e **Boltzmann** para múltiplos cenários.")

    # --- SIDEBAR ---
    st.sidebar.header("Configuração")
    uploaded_file = st.sidebar.file_uploader("Arquivo de Dados", type=["csv", "xlsx"])
    
    # Controle de velocidade
    effort = st.sidebar.select_slider("Qualidade do Ajuste (Iterações)", 
                                      options=["Rápido (Draft)", "Normal", "Preciso"], 
                                      value="Rápido (Draft)")
    iter_map = {"Rápido (Draft)": 100, "Normal": 300, "Preciso": 600}
    max_iter = iter_map[effort]

    if not uploaded_file:
        st.info("Carregue um arquivo para começar.")
        st.stop()

    # Leitura
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro: {e}")
        st.stop()

    cols = df.columns.tolist()
    c1, c2 = st.sidebar.columns(2)
    t_col = c1.selectbox("Tempo", cols)
    y_col = c2.selectbox("Resposta (Y)", cols, index=1 if len(cols)>1 else 0)
    
    # Tratamento
    df[t_col] = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."), errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors='coerce')
    df = df.dropna(subset=[t_col, y_col])
    
    t_data = df[t_col].values
    y_data = df[y_col].values
    
    # Ordenar
    idx = np.argsort(t_data)
    t_data = t_data[idx]
    y_data = y_data[idx]
    
    st.sidebar.success(f"{len(t_data)} pontos carregados.")
    
    if not st.sidebar.button("RODAR TODAS AS ANÁLISES"):
        st.warning("Clique no botão na barra lateral para iniciar o processamento.")
        # Mostra gráfico cru apenas
        fig, ax = plt.subplots(figsize=(6,2))
        ax.scatter(t_data, y_data, s=10)
        ax.set_title("Dados Brutos")
        st.pyplot(fig)
        st.stop()

    # --- PROCESSAMENTO ---
    
    tab_gompertz, tab_boltzmann = st.tabs(["📊 Resultados Gompertz", "📈 Resultados Boltzmann"])
    
    # --- LOOP GOMPERTZ ---
    with tab_gompertz:
        st.subheader("Ajustes: Modelo Gompertz (Eq. 32)")
        best_aic_g = np.inf
        best_phase_g = 0
        
        # Barra de progresso dedicada
        prog_g = st.progress(0)
        
        for n in range(1, 6):
            # Header do Expander com indicador de status
            expander_title = f"{n} Fase(s)"
            with st.expander(expander_title, expanded=(n==1)):
                st.write(f"Calculando {n} fases...")
                
                # Roda o ajuste
                res = fit_single_scenario(
                    t_data, y_data, gompertz_phase_eq32, n_phases=n, 
                    de_maxiter=max_iter, seed=42
                )
                
                # Exibe
                display_result(res, t_data, y_data, gompertz_phase_eq32, n, "Gompertz")
                
                # Rastreia o melhor
                if res['aicc'] < best_aic_g:
                    best_aic_g = res['aicc']
                    best_phase_g = n
            
            prog_g.progress(n / 5)
            
        st.success(f"Melhor configuração Gompertz sugerida: **{best_phase_g} Fase(s)** (Menor AICc)")

    # --- LOOP BOLTZMANN ---
    with tab_boltzmann:
        st.subheader("Ajustes: Modelo Boltzmann (Eq. 31)")
        best_aic_b = np.inf
        best_phase_b = 0
        
        prog_b = st.progress(0)
        
        for n in range(1, 6):
            with st.expander(f"{n} Fase(s)", expanded=(n==1)):
                st.write(f"Calculando {n} fases...")
                
                res = fit_single_scenario(
                    t_data, y_data, boltzmann_phase_eq31, n_phases=n, 
                    de_maxiter=max_iter, seed=99 # Seed diferente para variar estocástica
                )
                
                display_result(res, t_data, y_data, boltzmann_phase_eq31, n, "Boltzmann")
                
                if res['aicc'] < best_aic_b:
                    best_aic_b = res['aicc']
                    best_phase_b = n
            
            prog_b.progress(n / 5)

        st.success(f"Melhor configuração Boltzmann sugerida: **{best_phase_b} Fase(s)** (Menor AICc)")

    # --- RESUMO FINAL ---
    st.divider()
    st.markdown("### Conclusão Estatística")
    if best_aic_g < best_aic_b:
        st.info(f"O modelo **Gompertz** com **{best_phase_g} fase(s)** apresentou o melhor ajuste estatístico global (AICc = {best_aic_g:.2f}).")
    else:
        st.info(f"O modelo **Boltzmann** com **{best_phase_b} fase(s)** apresentou o melhor ajuste estatístico global (AICc = {best_aic_b:.2f}).")

if __name__ == "__main__":
    main()
