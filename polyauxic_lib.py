"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 

"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import t as t_dist

# ==============================================================================
# 1. MODELOS MATEMÁTICOS
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
    y_i, y_f = theta[0], theta[1]
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)
    
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases

# ==============================================================================
# 2. FUNÇÕES DE PERDA E OTIMIZAÇÃO
# ==============================================================================

def sse_loss(theta, t, y, model_func, n_phases):
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    # Penalidade simples para evitar predições absurdas
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred) ** 2)

def robust_loss(theta, t, y, model_func, n_phases):
    """Perda Robusta (Soft L1) para deteção de outliers."""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    residuals = y - y_pred
    loss = 2.0 * (np.sqrt(1.0 + residuals**2) - 1.0)
    return np.sum(loss)

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
            'r_max': (np.max(y)-np.min(y))/(t_span/n_phases)
        })
    guesses.sort(key=lambda x: x['lambda'])
    
    theta_guess = np.zeros(2 + 3 * n_phases)
    n_slice = max(1, len(y)//5)
    mean_start, mean_end = np.mean(y[:n_slice]), np.mean(y[-n_slice:])
    if mean_start < mean_end:
         theta_guess[0], theta_guess[1] = np.min(y), np.max(y)
    else:
         theta_guess[0], theta_guess[1] = np.max(y), np.min(y)

    for i in range(n_phases):
        theta_guess[2 + n_phases + i] = guesses[i]['r_max']
        theta_guess[2 + 2 * n_phases + i] = guesses[i]['lambda']
    return theta_guess

def setup_bounds(n_phases, force_yi, force_yf):
    bounds = []
    bounds.append((0.0, 1e-10) if force_yi else (0.0, 1.5)) # yi
    bounds.append((0.0, 1e-10) if force_yf else (0.0, 2.0)) # yf
    for _ in range(n_phases): bounds.append((-10, 10))    # z
    for _ in range(n_phases): bounds.append((0.0, 500.0)) # r_max
    for _ in range(n_phases): bounds.append((0.0, 1.2))   # lambda
    return bounds

def fit_model_auto(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False, seed=42):
    """Ajuste final usando SSE (Mínimos Quadrados)."""
    np.random.seed(seed)
    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params: return None

    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm, y_norm = t_data / t_scale, y_data / y_scale

    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    
    # Normalizar chute inicial
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0 
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale

    bounds = setup_bounds(n_phases, force_yi, force_yf)
    
    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)
    if force_yi: init_pop[:, 0] = 0.0
    if force_yf: init_pop[:, 1] = 0.0

    # 1. Otimização Global
    res_de = differential_evolution(
        sse_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
        seed=seed, polish=True, tol=1e-6
    )

    # 2. Refinamento Local
    res_opt = minimize(
        sse_loss, res_de.x, args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B', bounds=bounds, tol=1e-10
    )

    # Rescale back
    theta_norm = res_opt.x
    theta_real = np.zeros_like(theta_norm)
    theta_real[0:2] = theta_norm[0:2] * y_scale
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_scale

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    sse = np.sum((y_data - y_pred) ** 2)
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12
    aicc = n_len * np.log(sse/n_len) + 2*k + (2*k*(k+1))/(n_len-k-1) if (n_len-k-1)>0 else np.inf

    return {"n_phases": n_phases, "theta": theta_real, "metrics": {"SSE": sse, "AICc": aicc}, "y_pred": y_pred}

def fit_model_auto_robust_pre(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False, seed=42):
    """Pré-ajuste Robusto (Soft L1) para detectar outliers."""
    np.random.seed(seed)
    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params: return None
    
    # Normalização e setup similar ao fit_model_auto, mas usando robust_loss
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm, y_norm = t_data / t_scale, y_data / y_scale

    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0 
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale

    bounds = setup_bounds(n_phases, force_yi, force_yf)
    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)
    if force_yi: init_pop[:, 0] = 0.0
    if force_yf: init_pop[:, 1] = 0.0

    res_de = differential_evolution(
        robust_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
        seed=seed, polish=True, tol=1e-6
    )
    
    res_opt = minimize(
        robust_loss, res_de.x, args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B', bounds=bounds, tol=1e-10
    )

    theta_norm = res_opt.x
    theta_real = np.zeros_like(theta_norm)
    theta_real[0:2] = theta_norm[0:2] * y_scale
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * (y_scale/t_scale)
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * t_scale

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    return {"theta": theta_real, "y_pred": y_pred}

def detect_outliers_rout_rigorous(y_true, y_pred, Q=1.0):
    """Detecção de outliers ROUT com FDR Q%."""
    residuals = y_true - y_pred
    n = residuals.size
    if n < 3: return np.zeros_like(residuals, dtype=bool)

    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    rsdr = 1.4826 * mad_res if mad_res > 1e-12 else 1e-12

    t_scores = residuals / rsdr
    df = max(n - 1, 1)
    p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_scores), df=df))

    alpha = Q / 100.0
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    i = np.arange(1, n + 1)
    bh_thresholds = (i / n) * alpha
    
    below = p_sorted <= bh_thresholds
    if not np.any(below): return np.zeros_like(residuals, dtype=bool)

    k_max = np.max(np.where(below)[0])
    p_crit = p_sorted[k_max]
    return p_values <= p_crit

def process_data(df):
    """Processa DataFrame para extrair pares t, y."""
    df = df.reset_index(drop=True)
    num_replicates = df.shape[1] // 2
    all_t, all_y = [], []
    for i in range(num_replicates):
        t = pd.to_numeric(df.iloc[:, 2*i], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 2*i+1], errors='coerce').values
        mask = ~np.isnan(t) & ~np.isnan(y)
        all_t.extend(t[mask])
        all_y.extend(y[mask])
    t_flat = np.array(all_t).flatten()
    y_flat = np.array(all_y).flatten()
    if len(t_flat) > 0:
        idx = np.argsort(t_flat)
        return t_flat[idx], y_flat[idx], []
    return np.array([]), np.array([]), []
