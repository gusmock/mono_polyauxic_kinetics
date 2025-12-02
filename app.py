import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. EQUAÇÕES MATEMÁTICAS (EQ. 31 e 32)
# ==============================================================================

def boltzmann_term_eq31(t, yi, yf, p_j, rmax_j, lam_j):
    """Modelo de Boltzmann (Eq. 31) - Termo da fase j"""
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = 4.0 * rmax_j * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, yi, yf, p_j, rmax_j, lam_j):
    """Modelo de Gompertz (Eq. 32) - Termo da fase j"""
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = rmax_j * np.e * (lam_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """Constrói o modelo completo somando as fases."""
    t = np.asarray(t, dtype=float)
    yi = theta[0]
    yf = theta[1]
    
    # Extração de parâmetros
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax (z -> p)
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * sum_phases

# ==============================================================================
# 2. ESTATÍSTICAS
# ==============================================================================

def calculate_information_criteria(y_true, y_pred, n_params):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    
    if sse <= 1e-12:
        return 0.0, 0.0, -np.inf, -np.inf, -np.inf

    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (sse / sst) if sst > 1e-12 else 0.0

    aic = n * np.log(sse / n) + 2 * n_params
    bic = n * np.log(sse / n) + n_params * np.log(n)
    
    if (n - n_params - 1) > 0:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
    else:
        aicc = np.inf

    return sse, r2, aic, aicc, bic

# ==============================================================================
# 3. ROTINA DE AJUSTE (COM SUPORTE A INPUT MANUAL)
# ==============================================================================

def objective_function(theta, t, y, model_func, n_phases):
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    return np.sum((y - y_pred)**2)

def fit_model(t_data, y_data, model_func, n_phases, max_iter=300, manual_guess=None):
    """
    Realiza o ajuste. 
    Se manual_guess for fornecido (dicionário com valores reais), pula o DE e usa esses valores
    como estimativa inicial para o minimizador local.
    """
    # 1. Normalização
    t_max = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_max = np.max(y_data) if np.max(y_data) > 0 else 1.0
    
    t_norm = t_data / t_max
    y_norm = y_data / y_max

    # 2. Bounds (para dados normalizados)
    bounds = []
    bounds.append((0.0, 1.5)) # yi
    bounds.append((0.0, 2.0)) # yf
    for _ in range(n_phases): bounds.append((-10, 10)) # z
    for _ in range(n_phases): bounds.append((0, 100.0)) # rmax
    for _ in range(n_phases): bounds.append((0, 1.2))   # lam

    # 3. Definição do Ponto de Partida (x0)
    
    # Caso A: Usuário definiu parâmetros manuais (apenas para 3 fases, mas a lógica é geral)
    if manual_guess is not None:
        # Converter os valores REAIS do usuário para valores NORMALIZADOS do otimizador
        theta0_norm = np.zeros(2 + 3*n_phases)
        
        # yi, yf
        theta0_norm[0] = manual_guess['yi'] / y_max
        theta0_norm[1] = manual_guess['yf'] / y_max
        
        # z (Converter frações p em logits z)
        # Assumimos z = ln(p). O Softmax cuidará da normalização.
        p_user = np.array(manual_guess['p'])
        # Pequena proteção contra log(0)
        p_user = np.maximum(p_user, 1e-6)
        z_user = np.log(p_user)
        theta0_norm[2 : 2+n_phases] = z_user
        
        # rmax (Normalizado) -> r_norm = r_real / (y_max/t_max)
        r_factor = y_max / t_max
        r_user = np.array(manual_guess['rmax'])
        theta0_norm[2+n_phases : 2+2*n_phases] = r_user / r_factor
        
        # lam (Normalizado) -> lam_norm = lam_real / t_max
        l_user = np.array(manual_guess['lam'])
        theta0_norm[2+2*n_phases : 2+3*n_phases] = l_user / t_max
        
        # Definimos x0 direto, pulando a Evolução Diferencial
        x0 = theta0_norm
        
    # Caso B: Automático (Evolução Diferencial)
    else:
        result_global = differential_evolution(
            objective_function,
            bounds,
            args=(t_norm, y_norm, model_func, n_phases),
            maxiter=max_iter,
            popsize=20,
            strategy='best1bin',
            seed=42,
            polish=False
        )
        x0 = result_global.x

    # 4. Refinamento Local (Minimização)
    # Mesmo com input manual, rodamos isso para "ajustar fino" os dados aos pontos exatos
    result_local = minimize(
        objective_function,
        x0,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    theta_norm = result_local.x
    
    # 5. Desnormalização (Volta para escala real)
    theta_real = np.zeros_like(theta_norm)
    theta_real[0] = theta_norm[0] * y_max
    theta_real[1] = theta_norm[1] * y_max
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases] # z
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_max / t_max)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_max

    # 6. Métricas Finais
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    sse, r2, aic, aicc, bic = calculate_information_criteria(y_data, y_pred, len(theta_real))
    
    return {
        "theta": theta_real,
        "sse": sse, "r2": r2, "aic": aic, "bic": bic, "aicc": aicc,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. INTERFACE GRÁFICA
# ==============================================================================

def display_results_block(res, n_phases, color_code, t_data, y_data, model_func):
    """Helper para exibir gráficos e tabelas limpos."""
    theta = res["theta"]
    yi, yf = theta[0], theta[1]
    
    # Processar parâmetros das fases
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]
    
    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))
    
    phases = [{"p": p[i], "rmax": rmax[i], "lam": lam[i]} for i in range(n_phases)]
    phases.sort(key=lambda x: x["lam"])
    
    # Layout: Gráfico na esquerda, Tabelas na direita
    c_graph, c_table = st.columns([1.5, 1])
    
    with c_graph:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(t_data, y_data, color='black', alpha=0.5, s=40, label='Experimental')
        
        t_smooth = np.linspace(min(t_data), max(t_data), 300)
        y_global = polyauxic_model(t_smooth, theta, model_func, n_phases)
        ax.plot(t_smooth, y_global, color=color_code, linewidth=2.5, label='Modelo Global')
        
        # Fases individuais
        colors_ph = plt.cm.viridis(np.linspace(0, 0.9, n_phases))
        for i, ph in enumerate(phases):
            y_indiv = model_func(t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
            y_vis = yi + (yf - yi) * y_indiv
            ax.plot(t_smooth, y_vis, '--', color=colors_ph[i], alpha=0.8, linewidth=1.5, label=f'Fase {i+1}')
            
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Produto / Crescimento")
        st.pyplot(fig)
        
    with c_table:
        st.markdown("**Métricas de Qualidade**")
        met_df = pd.DataFrame({
            "Métrica": ["R²", "AICc", "SSE"],
            "Valor": [res['r2'], res['aicc'], res['sse']]
        })
        st.dataframe(met_df.style.format({"Valor": "{:.4f}"}), hide_index=True)
        
        st.markdown("**Parâmetros Globais**")
        st.write(f"**y_i:** {yi:.4f} | **y_f:** {yf:.4f}")
        
        st.markdown("**Parâmetros das Fases**")
        df_ph = pd.DataFrame(phases)
        df_ph.index += 1
        df_ph = df_ph.rename(columns={"p": "Fração", "rmax": "Taxa Max", "lam": "Latência"})
        st.dataframe(df_ph.style.format("{:.4f}"))


def main():
    st.set_page_config(page_title="Poliauxico Híbrido (Auto/Manual)", layout="wide")
    st.title("Modelagem Poliauxica Híbrida")
    st.markdown("Ajuste automático para 1, 2, 4, 5 fases. **Ajuste manual opcional para 3 fases.**")

    # --- SIDEBAR: DADOS ---
    st.sidebar.header("1. Carregar Dados")
    file = st.sidebar.file_uploader("Arquivo (CSV/XLSX)", type=["csv", "xlsx"])
    
    st.sidebar.divider()
    
    # --- SIDEBAR: INPUT MANUAL 3 FASES ---
    st.sidebar.header("2. Configuração Manual (3 Fases)")
    use_manual_3 = st.sidebar.checkbox("Definir estimativas para 3 Fases?", value=False)
    
    manual_params_3 = None
    
    if use_manual_3:
        with st.sidebar.expander("Parâmetros Iniciais (3 Fases)", expanded=True):
            st.caption("Insira valores aproximados (reais) para guiar o otimizador.")
            m_yi = st.number_input("y_i (Inicial)", value=0.0, format="%.4f")
            m_yf = st.number_input("y_f (Final)", value=1.0, format="%.4f")
            
            st.markdown("---")
            m_p1 = st.number_input("Fração Fase 1 (p1)", 0.0, 1.0, 0.33)
            m_r1 = st.number_input("Taxa Fase 1 (rmax1)", 0.0, 100.0, 1.0)
            m_l1 = st.number_input("Latência Fase 1 (λ1)", 0.0, 1000.0, 5.0)
            
            st.markdown("---")
            m_p2 = st.number_input("Fração Fase 2 (p2)", 0.0, 1.0, 0.33)
            m_r2 = st.number_input("Taxa Fase 2 (rmax2)", 0.0, 100.0, 1.0)
            m_l2 = st.number_input("Latência Fase 2 (λ2)", 0.0, 1000.0, 15.0)

            st.markdown("---")
            m_p3 = st.number_input("Fração Fase 3 (p3)", 0.0, 1.0, 0.34)
            m_r3 = st.number_input("Taxa Fase 3 (rmax3)", 0.0, 100.0, 1.0)
            m_l3 = st.number_input("Latência Fase 3 (λ3)", 0.0, 1000.0, 25.0)
            
            # Normalizar frações para somar 1 (opcional, mas bom pra consistência)
            total_p = m_p1 + m_p2 + m_p3
            if total_p == 0: total_p = 1
            
            manual_params_3 = {
                'yi': m_yi, 'yf': m_yf,
                'p': [m_p1/total_p, m_p2/total_p, m_p3/total_p],
                'rmax': [m_r1, m_r2, m_r3],
                'lam': [m_l1, m_l2, m_l3]
            }

    # --- PROCESSAMENTO DOS DADOS ---
    if not file:
        st.info("Carregue um arquivo para começar.")
        st.stop()

    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except:
        st.error("Erro ao ler arquivo.")
        st.stop()
        
    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    t_col = c1.selectbox("Tempo", cols)
    y_col = c2.selectbox("Resposta", cols, index=1 if len(cols)>1 else 0)
    
    t_data = pd.to_numeric(df[t_col], errors='coerce').dropna().values
    y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().values
    
    # Garantir alinhamento
    min_len = min(len(t_data), len(y_data))
    t_data = t_data[:min_len]
    y_data = y_data[:min_len]
    
    # Ordenar
    idx = np.argsort(t_data)
    t_data, y_data = t_data[idx], y_data[idx]

    if st.button("Executar Modelagem (1 a 5 Fases)"):
        st.divider()
        
        tab_g, tab_b = st.tabs(["Gompertz (Eq. 32)", "Boltzmann (Eq. 31)"])
        
        # --- LOOP GOMPERTZ ---
        with tab_g:
            best_aic_g = np.inf
            for n in range(1, 6):
                # Define se usaremos o palpite manual
                guess = manual_params_3 if (n == 3 and use_manual_3) else None
                msg = f" (Manual)" if guess else " (Auto)"
                
                with st.expander(f"Ajuste {n} Fase(s){msg if n==3 else ''}", expanded=(n==3)):
                    res = fit_model(t_data, y_data, gompertz_term_eq32, n, manual_guess=guess)
                    display_results_block(res, n, "tab:blue", t_data, y_data, gompertz_term_eq32)
                    if res['aicc'] < best_aic_g: best_aic_g = res['aicc']

        # --- LOOP BOLTZMANN ---
        with tab_b:
            best_aic_b = np.inf
            for n in range(1, 6):
                guess = manual_params_3 if (n == 3 and use_manual_3) else None
                msg = f" (Manual)" if guess else " (Auto)"
                
                with st.expander(f"Ajuste {n} Fase(s){msg if n==3 else ''}", expanded=(n==3)):
                    res = fit_model(t_data, y_data, boltzmann_term_eq31, n, manual_guess=guess)
                    display_results_block(res, n, "tab:orange", t_data, y_data, boltzmann_term_eq31)
                    if res['aicc'] < best_aic_b: best_aic_b = res['aicc']

if __name__ == "__main__":
    main()
