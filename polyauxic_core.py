import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import t as t_dist

# ==============================================================================
# 1. MATHEMATICAL MODELS
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Boltzmann model term (Eq. 31)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Gompertz model term (Eq. 32)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """Global polyauxic model: Summation of weighted phases."""
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    
    # Softmax transformation for weights p
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)
    
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases

def sse_loss(theta, t, y, model_func, n_phases):
    """Sum of Squared Errors (SSE) Loss function."""
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    # Constraint: non-decreasing inflection points
    if np.any(np.diff(lambda_) <= 0):
        return 1e12

    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred) ** 2)

def robust_loss(theta, t, y, model_func, n_phases):
    """Soft L1 robust loss (used for ROUT pre-fit step)."""
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    if np.any(np.diff(lambda_) <= 0):
        return 1e12

    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    residuals = y - y_pred
    # Lorentzian-like loss
    loss = 2.0 * (np.sqrt(1.0 + residuals**2) - 1.0)
    return np.sum(loss)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Calculates Numerical Hessian for Standard Error estimation."""
    k = len(theta)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i = np.zeros(k)
            e_i[i] = epsilon
            e_j = np.zeros(k)
            e_j[j] = epsilon
            f_pp = func(theta + e_i + e_j, *args)
            f_pm = func(theta + e_i - e_j, *args)
            f_mp = func(theta - e_i + e_j, *args)
            f_mm = func(theta - e_i - e_j, *args)
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
    return hess

def detect_outliers(y_true, y_pred):
    """Simple outlier detection based on MAD (Z-score > 2.5)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def detect_outliers_rout_rigorous(y_true, y_pred, Q=1.0):
    """
    ROUT (Rigorous) Method with FDR control.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n = residuals.size
    if n < 3:
        return np.zeros_like(residuals, dtype=bool)

    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    rsdr = 1.4826 * mad_res if mad_res > 1e-12 else 1e-12

    t_scores = residuals / rsdr
    df = max(n - 1, 1)
    abs_t = np.abs(t_scores)
    p_values = 2.0 * (1.0 - t_dist.cdf(abs_t, df=df))

    alpha = Q / 100.0
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    i = np.arange(1, n + 1)
    bh_thresholds = (i / n) * alpha
    below = p_sorted <= bh_thresholds
    if not np.any(below):
        return np.zeros_like(residuals, dtype=bool)

    k_max = np.max(np.where(below)[0])
    p_crit = p_sorted[k_max]
    mask_outliers = p_values <= p_crit
    return mask_outliers

def smart_initial_guess(t, y, n_phases):
    """Heuristic initial parameter estimation using derivatives."""
    dy = np.gradient(y, t)
    dy_smooth = np.convolve(dy, np.ones(5) / 5, mode='same')
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth) * 0.1, distance=min_dist)
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lambda': t[p_idx], 'r_max': dy_smooth[p_idx]})
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lambda': t.min() + t_span * (len(guesses) + 1) / (n_phases + 1),
            'r_max': (np.max(y) - np.min(y)) / (t_span / n_phases)
        })
    guesses.sort(key=lambda x: x['lambda'])
    theta_guess = np.zeros(2 + 3 * n_phases)
    
    n_slice = max(1, len(y) // 5) 
    mean_start = np.mean(y[:n_slice])
    mean_end = np.mean(y[-n_slice:])
    
    if float(mean_start) < float(mean_end):
         theta_guess[0] = np.min(y)
         theta_guess[1] = np.max(y)
    else:
         theta_guess[0] = np.max(y)
         theta_guess[1] = np.min(y)

    theta_guess[2 : 2 + n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2 + n_phases + i] = guesses[i]['r_max']
        theta_guess[2 + 2 * n_phases + i] = guesses[i]['lambda']
    return theta_guess

def calculate_p_errors(z_vals, cov_z):
    """Standard error propagation for p (Softmax weights)."""
    exps = np.exp(z_vals - np.max(z_vals))
    p = exps / np.sum(exps)
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, j] = p[i] * (1 - p[i])
            else:
                J[i, j] = -p[i] * p[j]
    try:
        cov_p = J @ cov_z @ J.T
        se_p = np.sqrt(np.abs(np.diag(cov_p)))
        return se_p
    except:
        return np.full(n, np.nan)

# ==============================================================================
# 2. FITTING ENGINE
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False):
    """Main fitting function."""
    SEED_VALUE = 42
    np.random.seed(SEED_VALUE)

    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params:
        return None

    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale

    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    
    if force_yi:
        theta0_norm[0] = 0.0
    if force_yf:
        theta0_norm[1] = 0.0

    theta0_norm[2 : 2 + n_phases] = 0.0
    theta0_norm[2 + n_phases : 2 + 2 * n_phases] = theta_smart[2 + n_phases : 2 + 2 * n_phases] / (y_scale / t_scale)
    theta0_norm[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_smart[2 + 2 * n_phases : 2 + 3 * n_phases] / t_scale

    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    if force_yi:
        init_pop[:, 0] = 0.0
    if force_yf:
        init_pop[:, 1] = 0.0

    bounds = []
    # y_i bounds
    if force_yi:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 1.5)) 
    
    # y_f bounds
    if force_yf:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 2.0))
        
    for _ in range(n_phases):
        bounds.append((-10, 10))     # z
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))  # r_max_norm
    for _ in range(n_phases):
        bounds.append((0.0, 1.2))    # lambda_norm

    res_de = differential_evolution(
        sse_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=SEED_VALUE,
        polish=True,
        tol=1e-6
    )

    res_opt = minimize(
        sse_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-10
    )

    theta_norm = res_opt.x

    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)
    se_p = np.full(n_phases, np.nan)

    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2 : 2 + n_phases] = theta_norm[2 : 2 + n_phases]
    scale_r = y_scale / t_scale
    theta_real[2 + n_phases : 2 + 2 * n_phases] = theta_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
    scale_l = t_scale
    theta_real[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm) ** 2)
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2 : 2 + n_phases] = se_norm[2 : 2 + n_phases]
        se_real[2 + n_phases : 2 + 2 * n_phases] = se_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
        se_real[2 + 2 * n_phases : 2 + 3 * n_phases] = se_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

        idx_z_start = 2
        idx_z_end = 2 + n_phases
        cov_z = cov_norm[idx_z_start:idx_z_end, idx_z_start:idx_z_end]
        z_vals = theta_norm[idx_z_start:idx_z_end]
        se_p = calculate_p_errors(z_vals, cov_z)
    except:
        se_real = np.full_like(theta_real, np.nan)

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)

    sse = np.sum((y_data - y_pred) ** 2)
    sst = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - sse / sst
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12:
        sse = 1e-12
    if (n_len - k - 1) > 0:
        r2_adj = 1 - (1 - r2) * (n_len - 1) / (n_len - k - 1)
    else:
        r2_adj = np.nan
    aic = n_len * np.log(sse / n_len) + 2 * k
    bic = n_len * np.log(sse / n_len) + k * np.log(n_len)
    aicc = aic + (2 * k * (k + 1)) / (n_len - k - 1) if (n_len - k - 1) > 0 else np.inf

    return {
        "n_phases": n_phases,
        "theta": theta_real,
        "se": se_real,
        "se_p": se_p,
        "metrics": {"R2": r2, "R2_adj": r2_adj, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

def fit_model_auto_robust_pre(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False):
    """Robust pre-fit (Soft L1) used exclusively for outlier detection."""
    SEED_VALUE = 42
    np.random.seed(SEED_VALUE)

    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params:
        return None

    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale

    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    
    if force_yi:
        theta0_norm[0] = 0.0
    if force_yf:
        theta0_norm[1] = 0.0
        
    theta0_norm[2 : 2 + n_phases] = 0.0
    theta0_norm[2 + n_phases : 2 + 2 * n_phases] = theta_smart[2 + n_phases : 2 + 2 * n_phases] / (y_scale / t_scale)
    theta0_norm[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_smart[2 + 2 * n_phases : 2 + 3 * n_phases] / t_scale

    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    if force_yi:
        init_pop[:, 0] = 0.0
    if force_yf:
        init_pop[:, 1] = 0.0

    bounds = []
    if force_yi:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 1.5))
    
    if force_yf:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 2.0))
        
    for _ in range(n_phases):
        bounds.append((-10, 10))
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))
    for _ in range(n_phases):
        bounds.append((0.0, 1.2))

    res_de = differential_evolution(
        robust_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=SEED_VALUE,
        polish=True,
        tol=1e-6
    )

    res_opt = minimize(
        robust_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-10
    )

    theta_norm = res_opt.x

    theta_real = np.zeros_like(theta_norm)
    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2 : 2 + n_phases] = theta_norm[2 : 2 + n_phases]
    scale_r = y_scale / t_scale
    theta_real[2 + n_phases : 2 + 2 * n_phases] = theta_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
    scale_l = t_scale
    theta_real[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    return {"theta": theta_real, "y_pred": y_pred}

# ==============================================================================
# 3. DATA PROCESSING
# ==============================================================================

def process_data(df):
    """Processes DataFrame detecting replicates in pairs of columns."""
    df = df.reset_index(drop=True)
    num_cols = df.shape[1]
    num_replicates = num_cols // 2
    
    all_t = []
    all_y = []
    replicates = []
    
    for i in range(num_replicates):
        t_col_raw = df.iloc[:, 2 * i].values
        y_col_raw = df.iloc[:, 2 * i + 1].values
        
        t_vals = pd.to_numeric(t_col_raw, errors='coerce')
        y_vals = pd.to_numeric(y_col_raw, errors='coerce')
        
        if hasattr(t_vals, 'to_numpy'): t_vals = t_vals.to_numpy()
        if hasattr(y_vals, 'to_numpy'): y_vals = y_vals.to_numpy()
            
        t_vals = np.array(t_vals).flatten()
        y_vals = np.array(y_vals).flatten()
        
        mask = ~np.isnan(t_vals) & ~np.isnan(y_vals)
        t_clean = t_vals[mask]
        y_clean = y_vals[mask]
        
        all_t.extend(t_clean)
        all_y.extend(y_clean)
        replicates.append({'t': t_clean, 'y': y_clean, 'name': f'Replica {i + 1}'})
        
    t_flat = np.array(all_t).flatten()
    y_flat = np.array(all_y).flatten()
    
    if len(t_flat) > 0:
        idx_sort = np.argsort(t_flat)
        return t_flat[idx_sort], y_flat[idx_sort], replicates
    else:
        return np.array([]), np.array([]), []

def calculate_mean_with_outliers(replicates, model_func, theta, n_phases):
    """Calculates mean statistics excluding detected outliers."""
    all_data = []
    for rep in replicates:
        for t, y in zip(rep['t'], rep['y']):
            all_data.append({'t': t, 'y': y})
    df_all = pd.DataFrame(all_data)
    y_pred_all = polyauxic_model(df_all['t'].values, theta, model_func, n_phases)
    outliers_mask = detect_outliers(df_all['y'].values, y_pred_all)
    df_all['is_outlier'] = outliers_mask
    df_all['t_round'] = df_all['t'].round(4)
    grouped = df_all[~df_all['is_outlier']].groupby('t_round')['y'].agg(['mean', 'std']).reset_index()
    return grouped, df_all

# ==============================================================================
# 5. INFORMATION CRITERIA SELECTION HELPERS
# ==============================================================================

def choose_information_criterion(N, k_max):
    """Selects AIC, AICc or BIC based on sample size N and parameters k."""
    dof_ratio = N / max(k_max, 1)
    if N <= 200:
        if dof_ratio < 40:
            return "AICc"
        else:
            return "AIC"
    else:
        return "BIC"

def select_first_local_min_index(values, tol=1e-9):
    """Selects the first local minimum index."""
    if not values:
        return 0
    best_idx = 0
    for i in range(1, len(values)):
        if values[i] < values[best_idx] - tol:
            best_idx = i
        elif values[i] >= values[best_idx] - tol:
            break
    return best_idx
