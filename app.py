import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. MODELOS SIGMOIDAIS POLIAUXICOS (Eqs. 31 e 32 – METODOLOGIA OFICIAL)
# ==============================================================================

def boltzmann_phase_eq31(t, yi, yf, p, rmax, lam):
    """
    Termo j da Eq. 31 (Modelo Boltzmann Poliauxico):
    y(x) = y_i + (y_f - y_i) * Σ p_j / (1 + exp( 4 r_j^max (λ_j - x)/((y_f-y_i)p_j) + 2 ))
    """
    delta_y = max(yf - yi, 1e-8)
    p_safe = max(p, 1e-12)

    # O expoente corresponde ao argumento da exponencial no denominador
    exponent = 4.0 * rmax * (lam - t) / (delta_y * p_safe) + 2.0
    exponent = np.clip(exponent, -100.0, 100.0)

    return p_safe / (1.0 + np.exp(exponent))


def gompertz_phase_eq32(t, yi, yf, p, rmax, lam):
    """
    Termo j da Eq. 32 (Modelo Gompertz Poliauxico):
    y(x) = y_i + (y_f - y_i) * Σ p_j exp( -exp( 1 + r_j^max e (λ_j - x)/((y_f-y_i)p_j) ) )
    """
    delta_y = max(yf - yi, 1e-8)
    p_safe = max(p, 1e-12)

    # O expoente interno da dupla exponencial
    exponent = 1.0 + (rmax * np.e) * (lam - t) / (delta_y * p_safe)
    exponent = np.clip(exponent, -100.0, 100.0)

    inner = np.exp(exponent)
    return p_safe * np.exp(-inner)


def polyauxic_model(t, theta, phase_func, n_phases):
    """
    Modelo completo combinando n fases.
    
    Parâmetros theta:
      [ y_i, y_f, z_1..z_n, r_1..r_n, λ_1..λ_n ]
    
    Onde p_j = softmax(z_j)
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)

    yi = theta[0]
    yf = theta[1]

    n = n_phases

    # Fatiamento dos parâmetros
    z = theta[2 : 2 + n]
    rmax = theta[2 + n : 2 + 2*n]
    lam = theta[2 + 2*n : 2 + 3*n]

    # Cálculo dos pesos p_j via softmax para garantir soma = 1
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # Somatório das fases
    y_red = 0.0
    for j in range(n):
        y_red += phase_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * y_red


# ==============================================================================
# 2. FUNÇÕES DE PERDA E ESTATÍSTICAS
# ==============================================================================

def lorentzian_loss(params, t_exp, y_exp, model_func, n_phases):
    """
    Função de perda Lorentziana para robustez contra outliers (M-estimator).
    """
    y_pred = polyauxic_model(t_exp, params, model_func, n_phases)
    residuals = y_exp - y_pred

    # Escala robusta baseada no MAD (Median Absolute Deviation)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = 1.4826 * mad if mad > 1e-12 else 1.0

    # Soma dos logaritmos da Lorentziana
    return np.sum(np.log(1.0 + (residuals / scale)**2))


def calculate_sse_r2(y_true, y_pred):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - sse/ss_tot if ss_tot > 0 else 0.0
    return sse, r2


def information_criteria(y_true, y_pred, n_params):
    n_data = len(y_true)
    sse, _ = calculate_sse_r2(y_true, y_pred)

    if sse <= 0:
        return np.inf, np.inf, np.inf, sse

    # Log-likelihood assumindo erros normais (aproximação para cálculo de IC)
    logL = -n_data/2 * (np.log(2*np.pi*sse/n_data) + 1)
    
    AIC = 2*n_params - 2*logL
    
    if n_data - n_params - 1 > 0:
        AICc = AIC + (2*n_params*(n_params+1)) / (n_data - n_params - 1)
    else:
        AICc = np.inf

    BIC = n_params * np.log(n_data) - 2*logL

    return AIC, AICc, BIC, sse


def select_ic_value(n_data, n_params, AIC, AICc, BIC):
    """
    Seleção automática do critério de informação baseada no tamanho da amostra.
    """
    if n_data > 200:
        return BIC, "BIC"
    if n_data / n_params < 40:
        return AICc, "AICc"
    return AIC, "AIC"


# ==============================================================================
# 3. HESSIANA E ERROS-PADRÃO
# ==============================================================================

def numerical_hessian(func, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps

            fpp = func(x + ei + ej)
            fpm = func(x + ei - ej)
            fmp = func(x - ei + ej)
            fmm = func(x - ei - ej)

            H[i,j] = (fpp - fpm - fmp + fmm)/(4*eps**2)

    return H


def parameter_uncertainty(theta_hat, t_data, y_data, model_func, n_phases):
    n_data = len(y_data)
    n_params = len(theta_hat)

    y_pred = polyauxic_model(t_data, theta_hat, model_func, n_phases)
    sse, _ = calculate_sse_r2(y_data, y_pred)

    if n_data <= n_params:
        return np.full_like(theta_hat, np.nan)

    sigma2 = sse / (n_data - n_params)

    def loss_fn(p):
        return lorentzian_loss(p, t_data, y_data, model_func, n_phases)

    try:
        H = numerical_hessian(loss_fn, theta_hat)
        # Matriz de covariância aproximada
        C = sigma2 * np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(C)))
    except Exception:
        se = np.full_like(theta_hat, np.nan)

    return se


# ==============================================================================
# 4. AJUSTE AUTOMÁTICO
# ==============================================================================

def fit_model_family(t_data, y_data, model_func, model_name, max_phases=4,
                     de_maxiter=500, random_seed=42):
    
    rng = np.random.default_rng(random_seed)
    n_data = len(y_data)
    y_min = np.min(y_data)
    y_max = np.max(y_data)
    t_max = np.max(t_data)

    best_result = None
    Cr = np.inf  # Melhor critério encontrado até agora

    # Loop para testar 1 até max_phases fases
    for n in range(1, max_phases+1):
        bounds = []
        
        # Bounds para y_i e y_f
        bounds.append((0, y_max))       # yi
        bounds.append((0, 1.5*y_max))   # yf

        # Bounds para z_j (softmax parameters)
        for _ in range(n):
            bounds.append((-10, 10))

        # Bounds para r_j^max
        for _ in range(n):
            bounds.append((0, (y_max/max(t_max,1e-6))*10))

        # Bounds para lambda_j
        for _ in range(n):
            bounds.append((0, t_max))

        # 1. Otimização Global (Differential Evolution)
        global_res = differential_evolution(
            lorentzian_loss,
            bounds,
            args=(t_data, y_data, model_func, n),
            maxiter=de_maxiter,
            popsize=15,
            tol=0.01,
            strategy='best1bin',
            seed=int(rng.integers(0, 1_000_000)),
            polish=False
        )

        theta_global = global_res.x

        # 2. Refinamento Local (Nelder-Mead)
        local_res = minimize(
            lorentzian_loss,
            theta_global,
            args=(t_data, y_data, model_func, n),
            method="Nelder-Mead",
            tol=1e-6
        )

        theta_hat = local_res.x
        y_pred = polyauxic_model(t_data, theta_hat, model_func, n)
        
        AIC, AICc, BIC, sse = information_criteria(y_data, y_pred, len(theta_hat))
        ic_value, ic_used = select_ic_value(n_data, len(theta_hat), AIC, AICc, BIC)
        _, r2 = calculate_sse_r2(y_data, y_pred)

        # Seleção do modelo (Parcimónia)
        if ic_value < Cr:
            Cr = ic_value
            se = parameter_uncertainty(theta_hat, t_data, y_data, model_func, n)
            
            best_result = {
                "model_name": model_name,
                "model_func": model_func,
                "n_phases": n,
                "theta_hat": theta_hat,
                "theta_se": se,
                "loss": local_res.fun,
                "AIC": AIC,
                "AICc": AICc,
                "BIC": BIC,
                "ic_used": ic_used,
                "ic_value": ic_value,
                "sse": sse,
                "r2": r2
            }
        else:
            # Se adicionar fase piorou o critério (AIC/BIC), paramos por aqui
            break

    return best_result


# ==============================================================================
# 5. INTERFACE STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Poliauxico – Eq. 31 e 32", layout="wide")
    st.title("Ajuste Poliauxico (Boltzmann & Gompertz) – Metodologia Oficial")
    st.markdown("Implementação com Coerência Cronológica e Remoção Robusta de Outliers.")

    # --- SIDEBAR: DADOS ---
    st.sidebar.header("Dados")
    uploaded_file = st.sidebar.file_uploader("Arquivo CSV/XLSX", type=["csv","xlsx"])

    max_phases = st.sidebar.number_input("Nº máximo de fases", 1, 6, 4)
    de_maxiter = st.sidebar.number_input("Maxiter DE", 50, 2000, 500, 50)
    random_seed = st.sidebar.number_input("Seed", 0, 999999, 42)

    if not uploaded_file:
        st.info("Por favor, carregue um arquivo de dados para começar.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("O arquivo precisa ter pelo menos duas colunas (Tempo e Resposta).")
        st.stop()

    t_col = st.sidebar.selectbox("Coluna de Tempo", cols)
    y_col = st.sidebar.selectbox("Coluna de Resposta (y)", cols, index=1 if len(cols)>1 else 0)

    # Tratamento de dados (conversão string/float)
    t = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors="coerce")

    mask = ~np.isnan(t) & ~np.isnan(y)
    t_data = np.asarray(t[mask])
    y_data = np.asarray(y[mask])

    # Ordenar pelo tempo
    idx = np.argsort(t_data)
    t_data = t_data[idx]
    y_data = y_data[idx]

    st.sidebar.success(f"{len(t_data)} pontos válidos carregados.")

    # Gráfico Inicial
    fig_raw, ax_raw = plt.subplots(figsize=(8,4))
    ax_raw.scatter(t_data, y_data, alpha=0.7)
    ax_raw.set_xlabel("Tempo")
    ax_raw.set_ylabel("Resposta")
    ax_raw.set_title("Dados Experimentais")
    st.pyplot(fig_raw)

    if not st.button("Executar Ajuste"):
        st.stop()

    # --- EXECUÇÃO DOS AJUSTES ---
    progress_bar = st.progress(0)
    st.write("Ajustando Gompertz (Eq. 32)...")
    gompertz_best = fit_model_family(
        t_data, y_data, gompertz_phase_eq32,
        "Gompertz (Eq. 32)", max_phases, de_maxiter, random_seed
    )
    progress_bar.progress(50)

    st.write("Ajustando Boltzmann (Eq. 31)...")
    boltzmann_best = fit_model_family(
        t_data, y_data, boltzmann_phase_eq31,
        "Boltzmann (Eq. 31)", max_phases, de_maxiter, random_seed+1
    )
    progress_bar.progress(100)

    if gompertz_best is None or boltzmann_best is None:
        st.error("Falha na convergência dos modelos.")
        st.stop()

    # --- COMPARAÇÃO E SELEÇÃO ---
    candidates = [gompertz_best, boltzmann_best]
    # Escolhe o menor Critério de Informação (IC)
    best = min(candidates, key=lambda d: d["ic_value"])

    st.divider()
    st.subheader("🏆 Melhor Modelo Encontrado")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo", best['model_name'])
    c2.metric("Nº Fases", best['n_phases'])
    c3.metric(f"Critério ({best['ic_used']})", f"{best['ic_value']:.4f}")
    c4.metric("R²", f"{best['r2']:.4f}")

    # --- PROCESSAMENTO DOS PARÂMETROS (COM ORDENAÇÃO) ---
    theta = best["theta_hat"]
    se_theta = best["theta_se"]
    n = best["n_phases"]

    yi = theta[0]
    yf = theta[1]
    
    # Extração dos parâmetros brutos (não ordenados)
    z_raw = theta[2 : 2+n]
    rmax_raw = theta[2+n : 2+2*n]
    lam_raw = theta[2+2*n : 2+3*n]

    # Cálculo dos pesos p antes de ordenar
    z_shift = z_raw - np.max(z_raw)
    p_raw = np.exp(z_shift) / np.sum(np.exp(z_shift))

    # Erros padrão (se disponíveis)
    yi_se = se_theta[0]
    yf_se = se_theta[1]
    
    # Tratamento seguro para erros NaN
    if np.any(np.isnan(se_theta)):
        rmax_se_raw = np.full(n, np.nan)
        lam_se_raw = np.full(n, np.nan)
    else:
        rmax_se_raw = se_theta[2+n : 2+2*n]
        lam_se_raw = se_theta[2+2*n : 2+3*n]

    # --- CRIAÇÃO DA ESTRUTURA ORDENADA (Chronological Coherence) ---
    # Criamos uma lista de dicionários para poder ordenar tudo junto baseado em Lambda
    phases_data = []
    for j in range(n):
        phases_data.append({
            "id_original": j,
            "p": p_raw[j],
            "rmax": rmax_raw[j],
            "lam": lam_raw[j],
            "rmax_se": rmax_se_raw[j],
            "lam_se": lam_se_raw[j]
        })

    # Ordena a lista pela latência (lambda)
    phases_data.sort(key=lambda x: x["lam"])

    # --- TABELAS DE RESULTADOS ---
    st.subheader("Parâmetros Globais")
    df_global = pd.DataFrame({
        "Parâmetro": ["y_i (Inicial)", "y_f (Final)"],
        "Valor Estimado": [yi, yf],
        "Erro Padrão": [yi_se, yf_se]
    })
    
    # CORREÇÃO: Aplicar formato apenas nas colunas numéricas
    st.table(df_global.style.format({
        "Valor Estimado": "{:.4f}",
        "Erro Padrão": "{:.4f}"
    }))

    st.subheader(f"Parâmetros das Fases (Ordenados Cronologicamente)")
    rows_table = []
    for idx_sorted, data in enumerate(phases_data):
        rows_table.append({
            "Fase": idx_sorted + 1,
            "Fração (p)": data["p"],
            "Taxa Max (rmax)": data["rmax"],
            "SE rmax": data["rmax_se"],
            "Latência (λ)": data["lam"],
            "SE λ": data["lam_se"]
        })
    
    df_phases = pd.DataFrame(rows_table)
    
    # CORREÇÃO: Aplicar formato específico nas colunas de float (ignorar 'Fase' que é int)
    st.table(df_phases.style.format({
        "Fração (p)": "{:.4f}",
        "Taxa Max (rmax)": "{:.4f}",
        "SE rmax": "{:.4f}",
        "Latência (λ)": "{:.4f}",
        "SE λ": "{:.4f}"
    }))

    # --- GRÁFICO FINAL (COM FASES ORDENADAS) ---
    st.subheader("Ajuste Gráfico")
    t_smooth = np.linspace(t_data.min(), t_data.max(), 400)
    
    # Para o modelo total, usamos o theta original (ordem não importa na soma total)
    y_pred_total = polyauxic_model(t_smooth, theta, best["model_func"], n)

    fig_b, ax_b = plt.subplots(figsize=(10, 6))
    
    # Dados Experimentais
    ax_b.scatter(t_data, y_data, color="black", s=40, alpha=0.6, label="Experimental")
    # Modelo Total
    ax_b.plot(t_smooth, y_pred_total, "k-", linewidth=2.5, label="Modelo Global")

    # Plotar fases individuais na ordem correta da legenda (Fase 1, 2, 3...)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]
    
    for idx_sorted, data in enumerate(phases_data):
        # Recalcula a curva da fase individual usando os parâmetros extraídos e ordenados
        y_red_phase = best["model_func"](
            t_smooth, yi, yf, 
            data["p"], data["rmax"], data["lam"]
        )
        # Escala para visualização (y_i até y_f)
        y_phase_vis = yi + (yf - yi) * y_red_phase
        
        color = colors[idx_sorted % len(colors)]
        ax_b.plot(t_smooth, y_phase_vis, "--", color=color, linewidth=1.5, 
                  label=f"Fase {idx_sorted+1} ($\lambda$={data['lam']:.1f})")

    ax_b.legend()
    ax_b.set_xlabel("Tempo")
    ax_b.set_ylabel("Resposta (y)")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_b)

    # --- RESUMO ESTATÍSTICO COMPARATIVO ---
    st.subheader("Comparativo de Modelos")
    
    def summarize(res):
        return {
            "Modelo": res["model_name"],
            "Fases": res["n_phases"],
            "Critério Usado": res["ic_used"],
            "Valor Critério": f"{res['ic_value']:.4f}",
            "R²": f"{res['r2']:.4f}",
            "AIC": f"{res['AIC']:.4f}",
            "BIC": f"{res['BIC']:.4f}",
            "SSE": f"{res['sse']:.4f}"
        }

    summary_df = pd.DataFrame([summarize(gompertz_best), summarize(boltzmann_best)])
    st.table(summary_df)

if __name__ == "__main__":
    main()
