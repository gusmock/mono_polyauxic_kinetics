import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. MODELOS MATEMÁTICOS (Eqs. 31 e 32)
# ==============================================================================

def boltzmann_phase_eq31(t, yi, yf, p, rmax, lam):
    """
    Fase individual Boltzmann (Eq. 31)
    """
    # Evita divisão por zero
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    
    p_safe = max(p, 1e-12)

    # Exponente: 4 * rmax * (lambda - t) / ((yf-yi)*p) + 2
    exponent = 4.0 * rmax * (lam - t) / (delta_y * p_safe) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0) # Clip para evitar overflow

    return p_safe / (1.0 + np.exp(exponent))


def gompertz_phase_eq32(t, yi, yf, p, rmax, lam):
    """
    Fase individual Gompertz (Eq. 32)
    """
    delta_y = yf - yi
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    
    p_safe = max(p, 1e-12)

    # Exponente interno
    exponent = 1.0 + (rmax * np.e) * (lam - t) / (delta_y * p_safe)
    exponent = np.clip(exponent, -500.0, 500.0)

    inner = np.exp(exponent)
    # Evita underflow no exp(-inner) se inner for muito grande
    return p_safe * np.exp(-inner)


def polyauxic_model(t, theta, phase_func, n_phases):
    """
    Modelo combinando n fases.
    theta: [yi, yf, z_1..z_n, rmax_1..rmax_n, lam_1..lam_n]
    """
    t = np.asarray(t, dtype=float)
    
    yi = theta[0]
    yf = theta[1]

    # Parametros das fases
    z = theta[2 : 2+n_phases]
    rmax = theta[2+n_phases : 2+2*n_phases]
    lam = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax para frações p_j
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # Soma das fases
    y_red = 0.0
    for j in range(n_phases):
        y_red += phase_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * y_red


# ==============================================================================
# 2. OTIMIZAÇÃO E FUNÇÃO OBJETIVO
# ==============================================================================

def sse_loss(params, t, y, model_func, n_phases):
    """Soma dos Erros Quadrados (SSE) - Garante maximização do R2"""
    y_pred = polyauxic_model(t, params, model_func, n_phases)
    residuals = y - y_pred
    return np.sum(residuals**2)

def calculate_stats(y_true, y_pred, n_params):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    
    r2 = 1 - (sse / sst) if sst > 1e-12 else 0.0
    
    n = len(y_true)
    # Critérios de Informação
    if sse <= 0:
        log_l = np.inf
    else:
        log_l = -n/2 * (np.log(2 * np.pi * sse / n) + 1)
    
    aic = 2*n_params - 2*log_l
    bic = n_params * np.log(n) - 2*log_l
    
    # AICc
    if n - n_params - 1 > 0:
        aicc = aic + (2*n_params*(n_params+1))/(n - n_params - 1)
    else:
        aicc = np.inf
        
    return sse, r2, aic, aicc, bic

# ==============================================================================
# 3. ROTINA DE AJUSTE (COM NORMALIZAÇÃO)
# ==============================================================================

def fit_model_normalized(t_data, y_data, model_func, model_name, max_phases, 
                         de_maxiter=200, random_seed=42):
    
    rng = np.random.default_rng(random_seed)
    
    # --- 1. NORMALIZAÇÃO DOS DADOS (CRUCIAL PARA CONVERGÊNCIA) ---
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    best_result = None
    best_ic = np.inf
    
    # Loop de Fases (1 até max_phases)
    for n in range(1, max_phases + 1):
        
        # --- DEFINIÇÃO DE BOUNDS PARA DADOS NORMALIZADOS ---
        # yi, yf (esperado entre 0 e 1.5 na escala normalizada)
        bounds = [(0, 1.2), (0, 1.5)] 
        
        # z (softmax)
        for _ in range(n): bounds.append((-5, 5))
        
        # rmax (taxa normalizada: inclinação na caixa 1x1. Geralmente < 30)
        for _ in range(n): bounds.append((0, 30.0))
        
        # lambda (tempo normalizado: 0 a 1)
        for _ in range(n): bounds.append((0, 1.1))
        
        # --- OTIMIZAÇÃO GLOBAL (DE) ---
        res_global = differential_evolution(
            sse_loss, 
            bounds,
            args=(t_norm, y_norm, model_func, n),
            maxiter=de_maxiter,
            popsize=20, # Aumentei popsize para melhor busca
            strategy='best1bin',
            seed=int(rng.integers(0, 100000)),
            polish=False,
            tol=0.001
        )
        
        # --- REFINAMENTO LOCAL ---
        res_local = minimize(
            sse_loss,
            res_global.x,
            args=(t_norm, y_norm, model_func, n),
            method='L-BFGS-B', # L-BFGS-B é ótimo para bounds
            bounds=bounds
        )
        
        theta_norm = res_local.x
        
        # --- DESNORMALIZAÇÃO DOS PARÂMETROS ---
        theta_real = np.zeros_like(theta_norm)
        
        # yi, yf
        theta_real[0] = theta_norm[0] * y_scale
        theta_real[1] = theta_norm[1] * y_scale
        
        # z (adimensionais, mantém)
        theta_real[2 : 2+n] = theta_norm[2 : 2+n]
        
        # rmax (unidade: y/t. Então r_real = r_norm * (yscale/tscale))
        theta_real[2+n : 2+2*n] = theta_norm[2+n : 2+2*n] * (y_scale / t_scale)
        
        # lambda (unidade: t. Então lam_real = lam_norm * tscale)
        theta_real[2+2*n : 2+3*n] = theta_norm[2+2*n : 2+3*n] * t_scale
        
        # --- CÁLCULO DE ESTATÍSTICAS REAIS ---
        y_pred = polyauxic_model(t_data, theta_real, model_func, n)
        sse, r2, aic, aicc, bic = calculate_stats(y_data, y_pred, len(theta_real))
        
        # Seleção do critério (AICc geralmente é o melhor padrão)
        ic_val = aicc
        
        # Lógica de seleção (Parcimônia)
        # Só aceita mais fases se reduzir o Critério significativamente (> 2 unidades)
        if ic_val < (best_ic - 2.0):
            best_ic = ic_val
            best_result = {
                "model_name": model_name,
                "model_func": model_func,
                "n_phases": n,
                "theta": theta_real,
                "sse": sse,
                "r2": r2,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "ic_val": ic_val
            }
        else:
            # Se adicionar fase não melhorou muito, para e mantém o anterior (mais simples)
            break
            
    return best_result


# ==============================================================================
# 4. INTERFACE
# ==============================================================================

def main():
    st.set_page_config(page_title="Poliauxico Robust", layout="wide")
    st.title("Ajuste Poliauxico - Versão Robusta (Normalizada)")
    
    st.sidebar.header("Configurações")
    uploaded_file = st.sidebar.file_uploader("Arquivo de Dados", type=["csv", "xlsx"])
    
    max_phases = st.sidebar.slider("Máximo de Fases", 1, 5, 4)
    effort = st.sidebar.select_slider("Esforço de Ajuste", options=["Rápido", "Normal", "Intenso"], value="Normal")
    
    iters_map = {"Rápido": 100, "Normal": 300, "Intenso": 1000}
    de_maxiter = iters_map[effort]

    if not uploaded_file:
        st.info("Aguardando arquivo...")
        st.stop()
        
    # Leitura dos dados
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except:
        st.error("Erro ao ler arquivo.")
        st.stop()
        
    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    t_col = c1.selectbox("Coluna Tempo", cols)
    y_col = c2.selectbox("Coluna Resposta (Y)", cols, index=1 if len(cols)>1 else 0)
    
    # Limpeza e Conversão
    df[t_col] = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."), errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors='coerce')
    df = df.dropna(subset=[t_col, y_col])
    
    t_data = df[t_col].values
    y_data = df[y_col].values
    
    # Ordenar por tempo
    idx = np.argsort(t_data)
    t_data = t_data[idx]
    y_data = y_data[idx]

    # Plot inicial
    fig_init, ax_init = plt.subplots(figsize=(8,3))
    ax_init.scatter(t_data, y_data, color='gray', alpha=0.6)
    ax_init.set_title("Pré-visualização dos Dados")
    st.pyplot(fig_init)
    
    if st.button("CALCULAR AJUSTE AGORA"):
        
        progress = st.progress(0)
        st.text("Otimizando Gompertz...")
        res_g = fit_model_normalized(t_data, y_data, gompertz_phase_eq32, "Gompertz", max_phases, de_maxiter, 42)
        progress.progress(50)
        
        st.text("Otimizando Boltzmann...")
        res_b = fit_model_normalized(t_data, y_data, boltzmann_phase_eq31, "Boltzmann", max_phases, de_maxiter, 99)
        progress.progress(100)
        
        # Comparação
        if res_g['aicc'] < res_b['aicc']:
            best = res_g
        else:
            best = res_b
            
        st.success(f"Modelo Vencedor: {best['model_name']} com {best['n_phases']} fases.")
        
        # --- EXIBIÇÃO ---
        theta = best['theta']
        n = best['n_phases']
        yi, yf = theta[0], theta[1]
        
        # Extração e Ordenação das Fases
        z = theta[2 : 2+n]
        rmax = theta[2+n : 2+2*n]
        lam = theta[2+2*n : 2+3*n]
        
        # Softmax
        z_shift = z - np.max(z)
        p = np.exp(z_shift)/np.sum(np.exp(z_shift))
        
        # Cria lista e ordena por lambda (tempo)
        phase_list = []
        for i in range(n):
            phase_list.append({
                "p": p[i],
                "rmax": rmax[i],
                "lam": lam[i]
            })
        phase_list.sort(key=lambda x: x['lam'])
        
        # Tabela Global
        st.subheader("Parâmetros Globais")
        st.dataframe(pd.DataFrame({
            "Parâmetro": ["y_i", "y_f", "R²", "AICc"],
            "Valor": [yi, yf, best['r2'], best['aicc']]
        }).style.format({"Valor": "{:.4f}"}))
        
        # Tabela Fases
        st.subheader("Parâmetros das Fases")
        rows = []
        for i, ph in enumerate(phase_list):
            rows.append({
                "Fase": i+1,
                "Fração (p)": ph['p'],
                "Taxa Max (rmax)": ph['rmax'],
                "Latência (λ)": ph['lam']
            })
        st.table(pd.DataFrame(rows).style.format({
            "Fração (p)": "{:.4f}",
            "Taxa Max (rmax)": "{:.4f}",
            "Latência (λ)": "{:.4f}"
        }))
        
        # Gráfico Final
        t_smooth = np.linspace(t_data.min(), t_data.max(), 300)
        y_smooth = polyauxic_model(t_smooth, theta, best['model_func'], n)
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(t_data, y_data, color='black', alpha=0.5, label='Dados')
        ax.plot(t_smooth, y_smooth, 'r-', linewidth=2, label='Modelo Global')
        
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for i, ph in enumerate(phase_list):
            # Plota contribuição individual visual
            # Nota: para visualização correta, usamos a função da fase isolada
            # escalada pela amplitude total
            y_indiv = best['model_func'](t_smooth, yi, yf, ph['p'], ph['rmax'], ph['lam'])
            y_vis = yi + (yf - yi) * y_indiv
            ax.plot(t_smooth, y_vis, '--', color=colors[i], label=f'Fase {i+1}')
            
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
