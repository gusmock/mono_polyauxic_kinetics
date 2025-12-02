import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. IMPLEMENTAÇÃO LITERAL DAS EQUAÇÕES 31 E 32
# ==============================================================================

def boltzmann_term_eq31(t, yi, yf, p_j, rmax_j, lam_j):
    """
    Calcula o termo individual de uma fase para o Modelo de Boltzmann (Eq. 31).
    
    Equação 31:
    y(x) = y_i + (y_f - y_i) * SOMATÓRIO [ p_j / (1 + exp( termo_exponencial )) ]
    
    Onde termo_exponencial = (4 * r_max * (lambda - x)) / ((yf - yi) * p_j) + 2
    """
    # Delta Y (y_f - y_i)
    delta_y = yf - yi
    
    # Proteção numérica para evitar divisão por zero
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Transcrição exata do argumento da exponencial da Eq. 31
    # Argumento = [ 4 * r_j^max * (λ_j - t) ] / [ (y_f - y_i) * p_j ] + 2
    numerator = 4.0 * rmax_j * (lam_j - t)
    denominator = delta_y * p_safe
    
    exponent = (numerator / denominator) + 2.0
    
    # Clip para estabilidade numérica (evitar overflow da exp)
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))


def gompertz_term_eq32(t, yi, yf, p_j, rmax_j, lam_j):
    """
    Calcula o termo individual de uma fase para o Modelo de Gompertz (Eq. 32).
    
    Equação 32:
    y(x) = y_i + (y_f - y_i) * SOMATÓRIO [ p_j * exp( -exp( termo_exponencial ) ) ]
    
    Onde termo_exponencial = (r_max * e * (lambda - x)) / ((yf - yi) * p_j) + 1
    """
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    # Transcrição exata do argumento da exponencial interna da Eq. 32
    # Argumento = [ r_j^max * e * (λ_j - t) ] / [ (y_f - y_i) * p_j ] + 1
    numerator = rmax_j * np.e * (lam_j - t)
    denominator = delta_y * p_safe
    
    exponent = (numerator / denominator) + 1.0
    
    # Clip para estabilidade numérica
    exponent = np.clip(exponent, -500.0, 500.0)

    # Estrutura Gompertz: exp( -exp( ... ) )
    return p_safe * np.exp(-np.exp(exponent))


def polyauxic_model(t, theta, model_func, n_phases):
    """
    Constrói o modelo completo somando as fases ponderadas.
    
    Parâmetros (theta):
      [0]: y_i
      [1]: y_f
      [2 ... 2+n]: z_j (parâmetros latentes para calcular p_j via softmax)
      [...]: r_max_j
      [...]: lambda_j
    """
    t = np.asarray(t, dtype=float)
    
    yi = theta[0]
    yf = theta[1]
    
    # Extração dos vetores de parâmetros
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    # Cálculo dos pesos p_j garantindo soma = 1 (Constraint)
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # Somatório das fases (Sigma da equação)
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, yi, yf, p[j], rmax[j], lam[j])

    # Equação Final: y_i + (y_f - y_i) * Somatório
    y_pred = yi + (yf - yi) * sum_phases
    
    return y_pred

# ==============================================================================
# 2. CÁLCULO DE ESTATÍSTICAS (AIC, BIC, SSE, R2)
# ==============================================================================

def calculate_information_criteria(y_true, y_pred, n_params):
    """
    Calcula AIC, AICc e BIC baseados na Soma dos Erros Quadrados (SSE).
    Assumindo erros normalmente distribuídos.
    """
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    
    # Evitar log de zero ou negativo
    if sse <= 1e-12:
        return 0.0, 0.0, -np.inf, -np.inf, -np.inf

    # R-quadrado
    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (sse / sst) if sst > 1e-12 else 0.0

    # Fórmula para AIC/BIC usando SSE (Least Squares Case)
    # AIC = n * ln(SSE/n) + 2*k
    aic = n * np.log(sse / n) + 2 * n_params
    
    # BIC = n * ln(SSE/n) + k * ln(n)
    bic = n * np.log(sse / n) + n_params * np.log(n)
    
    # AICc (Corrected)
    if (n - n_params - 1) > 0:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
    else:
        aicc = np.inf

    return sse, r2, aic, aicc, bic

# ==============================================================================
# 3. OTIMIZAÇÃO (DE + MINIMIZE)
# ==============================================================================

def objective_function(theta, t, y, model_func, n_phases):
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    # SSE Loss
    return np.sum((y - y_pred)**2)

def fit_model(t_data, y_data, model_func, n_phases, max_iter=500):
    # 1. Normalização dos dados para garantir convergência numérica
    t_max = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_max = np.max(y_data) if np.max(y_data) > 0 else 1.0
    
    t_norm = t_data / t_max
    y_norm = y_data / y_max

    # 2. Definição dos limites (Bounds) para variáveis normalizadas
    bounds = []
    bounds.append((0.0, 1.2)) # yi (normalizado)
    bounds.append((0.5, 1.5)) # yf (normalizado)
    
    # z (pesos softmax) - range amplo
    for _ in range(n_phases): bounds.append((-5, 5))
    # rmax (taxas) - range razoável para dados normalizados
    for _ in range(n_phases): bounds.append((0, 50.0))
    # lambda (latência) - deve estar dentro do tempo experimental
    for _ in range(n_phases): bounds.append((0, 1.0))

    # 3. Otimização Global (Evolution Strategy)
    result_global = differential_evolution(
        objective_function,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=max_iter,
        popsize=20,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        seed=42
    )

    # 4. Refinamento Local
    result_local = minimize(
        objective_function,
        result_global.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    theta_norm = result_local.x
    
    # 5. Desnormalização dos Parâmetros
    theta_real = np.zeros_like(theta_norm)
    
    # yi, yf
    theta_real[0] = theta_norm[0] * y_max
    theta_real[1] = theta_norm[1] * y_max
    # z (mantém)
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    # rmax = r_norm * (y_max / t_max)
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_max / t_max)
    # lambda = lam_norm * t_max
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_max

    # 6. Cálculo das Métricas Finais
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    sse, r2, aic, aicc, bic = calculate_information_criteria(y_data, y_pred, len(theta_real))
    
    return {
        "theta": theta_real,
        "sse": sse,
        "r2": r2,
        "aic": aic,
        "bic": bic,
        "aicc": aicc,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================

def display_phase_results(res, n_phases):
    """Exibe tabelas formatadas para os resultados."""
    theta = res["theta"]
    yi, yf = theta[0], theta[1]
    
    # Recuperar parâmetros das fases
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]
    
    # Calcular p real
    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))
    
    # Ordenação Cronológica (por Lambda)
    phases = []
    for i in range(n_phases):
        phases.append({"p": p[i], "rmax": rmax[i], "lam": lam[i]})
    phases.sort(key=lambda x: x["lam"])
    
    # 1. Tabela de Métricas
    st.markdown("##### Métricas de Ajuste")
    metrics_df = pd.DataFrame({
        "Métrica": ["R²", "SSE", "AIC", "BIC", "AICc"],
        "Valor": [res['r2'], res['sse'], res['aic'], res['bic'], res['aicc']]
    })
    st.dataframe(metrics_df.style.format({"Valor": "{:.4f}"}), hide_index=True)

    # 2. Tabela de Parâmetros Globais
    st.markdown("##### Parâmetros Globais")
    global_df = pd.DataFrame({
        "Parâmetro": ["y_i (Inicial)", "y_f (Final)"],
        "Valor": [yi, yf]
    })
    st.dataframe(global_df.style.format({"Valor": "{:.4f}"}), hide_index=True)
    
    # 3. Tabela de Fases
    st.markdown("##### Parâmetros das Fases (Ordenado por Tempo)")
    phase_rows = []
    for i, ph in enumerate(phases):
        phase_rows.append({
            "Fase": i+1,
            "Fração (p)": ph['p'],
            "Taxa (rmax)": ph['rmax'],
            "Latência (λ)": ph['lam']
        })
    phase_df = pd.DataFrame(phase_rows)
    
    # Formatação segura
    st.dataframe(phase_df.style.format({
        "Fração (p)": "{:.4f}",
        "Taxa (rmax)": "{:.4f}",
        "Latência (λ)": "{:.4f}"
    }), hide_index=True)


def plot_fit(t_data, y_data, res, model_func, n_phases, color_main):
    """Gera o gráfico do ajuste."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Dados reais
    ax.scatter(t_data, y_data, color='black', alpha=0.6, s=30, label='Dados Experimentais')
    
    # Modelo ajustado
    t_smooth = np.linspace(min(t_data), max(t_data), 300)
    y_smooth = polyauxic_model(t_smooth, res['theta'], model_func, n_phases)
    ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label='Ajuste Global')
    
    # Plotar fases individuais (Visualização)
    theta = res["theta"]
    yi, yf = theta[0], theta[1]
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]
    z_shift = z - np.max(z)
    p = np.exp(z_shift) / np.sum(np.exp(z_shift))
    
    # Agrupar e ordenar para plotar legenda correta (Fase 1 = primeira a acontecer)
    phases_raw = [{"p": p[i], "rmax": rmax[i], "lam": lam[i]} for i in range(n_phases)]
    phases_raw.sort(key=lambda x: x["lam"])
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(phases_raw)))
    for i, ph in enumerate(phases_raw):
        # Calcula a curva da fase isolada
        y_indiv = model_func(t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
        # Escala para a dimensão real do gráfico
        y_vis = yi + (yf - yi) * y_indiv
        ax.plot(t_smooth, y_vis, '--', color=colors[i], linewidth=1.5, label=f'Fase {i+1}')

    ax.set_xlabel("Tempo")
    ax.set_ylabel("Crescimento / Produto")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.4)
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Cinética Poliauxica Exata", layout="wide")
    st.title("Ajuste Cinético Poliauxico")
    st.markdown("""
    **Metodologia Oficial:** Ajuste simultâneo das Equações 31 (Boltzmann) e 32 (Gompertz).
    **Critérios:** R², SSE, AIC, BIC e AICc.
    """)

    # Upload
    st.sidebar.header("Dados de Entrada")
    file = st.sidebar.file_uploader("Arquivo CSV ou Excel", type=["csv", "xlsx"])
    
    max_iter = st.sidebar.slider("Iterações (Esforço)", 100, 1000, 300)
    
    if not file:
        st.info("Aguardando arquivo de dados...")
        st.stop()
        
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except:
        st.error("Erro ao ler arquivo.")
        st.stop()

    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    t_col = c1.selectbox("Tempo (t)", cols)
    y_col = c2.selectbox("Variável (y)", cols, index=1 if len(cols)>1 else 0)
    
    # Tratamento de dados
    df[t_col] = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."), errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors='coerce')
    df = df.dropna(subset=[t_col, y_col])
    
    t_data = df[t_col].values
    y_data = df[y_col].values
    idx = np.argsort(t_data)
    t_data, y_data = t_data[idx], y_data[idx]

    if st.button("Executar Ajustes (1 a 5 fases)"):
        
        tab1, tab2 = st.tabs(["Modelo Gompertz (Eq. 32)", "Modelo Boltzmann (Eq. 31)"])
        
        # --- GOMPERTZ ---
        with tab1:
            st.subheader("Resultados: Gompertz")
            progress_g = st.progress(0)
            best_aic_g = np.inf
            
            for n in range(1, 6):
                with st.expander(f"Ajuste com {n} Fase(s)", expanded=(n==1)):
                    res = fit_model(t_data, y_data, gompertz_term_eq32, n, max_iter)
                    
                    c_plot, c_data = st.columns([1.5, 1])
                    with c_plot:
                        plot_fit(t_data, y_data, res, gompertz_term_eq32, n, 'tab:blue')
                    with c_data:
                        display_phase_results(res, n)
                        
                    if res['aicc'] < best_aic_g: best_aic_g = res['aicc']
                progress_g.progress(n*20)

        # --- BOLTZMANN ---
        with tab2:
            st.subheader("Resultados: Boltzmann")
            progress_b = st.progress(0)
            best_aic_b = np.inf
            
            for n in range(1, 6):
                with st.expander(f"Ajuste com {n} Fase(s)", expanded=(n==1)):
                    res = fit_model(t_data, y_data, boltzmann_term_eq31, n, max_iter)
                    
                    c_plot, c_data = st.columns([1.5, 1])
                    with c_plot:
                        plot_fit(t_data, y_data, res, boltzmann_term_eq31, n, 'tab:orange')
                    with c_data:
                        display_phase_results(res, n)
                    
                    if res['aicc'] < best_aic_b: best_aic_b = res['aicc']
                progress_b.progress(n*20)
        
        st.success("Cálculos finalizados!")

if __name__ == "__main__":
    main()
