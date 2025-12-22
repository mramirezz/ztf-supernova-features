"""
Ajuste con MCMC usando emcee
"""
import numpy as np
import emcee
from typing import Dict, Tuple
from model import alerce_model
from config import MCMC_CONFIG, MODEL_CONFIG

def log_likelihood(params, times, flux, flux_err, dynamic_bounds=None):
    """
    Log-likelihood para MCMC
    """
    A, f, t0, t_rise, t_fall, gamma = params
    
    # Usar bounds dinámicos si se proporcionan, sino usar los de config
    if dynamic_bounds is None:
        bounds = MODEL_CONFIG["bounds"]
    else:
        bounds = dynamic_bounds
    
    if not (bounds["A"][0] < A < bounds["A"][1]):
        return -np.inf
    if not (bounds["f"][0] < f < bounds["f"][1]):
        return -np.inf
    if not (bounds["t0"][0] < t0 < bounds["t0"][1]):
        return -np.inf
    if not (bounds["t_rise"][0] < t_rise < bounds["t_rise"][1]):
        return -np.inf
    if not (bounds["t_fall"][0] < t_fall < bounds["t_fall"][1]):
        return -np.inf
    if not (bounds["gamma"][0] < gamma < bounds["gamma"][1]):
        return -np.inf
    
    try:
        model_flux = alerce_model(times, A, f, t0, t_rise, t_fall, gamma)
        
        # Verificar que el modelo sea válido
        if np.any(np.isnan(model_flux)) or np.any(model_flux <= 0):
            return -np.inf
        
        # Chi-cuadrado
        chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
        return -0.5 * chi2
    except:
        return -np.inf

def log_prior(params, dynamic_bounds=None):
    """
    Prior uniforme dentro de los bounds
    """
    A, f, t0, t_rise, t_fall, gamma = params
    
    # Usar bounds dinámicos si se proporcionan, sino usar los de config
    if dynamic_bounds is None:
        bounds = MODEL_CONFIG["bounds"]
    else:
        bounds = dynamic_bounds
    
    if (bounds["A"][0] < A < bounds["A"][1] and
        bounds["f"][0] < f < bounds["f"][1] and
        bounds["t0"][0] < t0 < bounds["t0"][1] and
        bounds["t_rise"][0] < t_rise < bounds["t_rise"][1] and
        bounds["t_fall"][0] < t_fall < bounds["t_fall"][1] and
        bounds["gamma"][0] < gamma < bounds["gamma"][1]):
        return 0.0
    return -np.inf

def log_posterior(params, times, flux, flux_err, dynamic_bounds=None):
    """
    Log posterior = log prior + log likelihood
    """
    lp = log_prior(params, dynamic_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, times, flux, flux_err, dynamic_bounds)

def fit_mcmc(phase, flux, flux_err, p0=None, verbose=True):
    """
    Ajustar modelo con MCMC
    
    Parameters:
    -----------
    phase, flux, flux_err : arrays
        Datos de la curva de luz
    p0 : array, optional
        Valores iniciales [A, f, t0, t_rise, t_fall, gamma]
    verbose : bool
        Mostrar progreso
        
    Returns:
    --------
    results : dict
        Parámetros, errores, samples, etc.
    """
    if p0 is None:
        from model import get_initial_guess
        p0 = get_initial_guess(phase, flux)
    
    # Ajustar bounds de t0 dinámicamente basado en el rango de fase
    # t0 debe estar dentro del rango de fase observado, con un margen
    phase_min = phase.min()
    phase_max = phase.max()
    phase_range = phase_max - phase_min
    
    # t0 puede estar un poco antes del inicio o después del final
    # para capturar el pico si está cerca de los bordes
    t0_margin = max(50.0, phase_range * 0.2)  # 20% del rango o mínimo 50 días
    t0_bounds = (phase_min - t0_margin, phase_max + t0_margin)
    
    # Ajustar bounds de A dinámicamente basado en el flujo observado
    # A debe ser razonable comparado con el flujo observado
    flux_min = flux.min()
    flux_max = flux.max()
    flux_median = np.median(flux)
    flux_std = np.std(flux)
    
    # A debe estar en un rango razonable: desde un poco menos que el mínimo observado
    # hasta varias veces el máximo observado (para permitir picos no observados)
    # Pero no más allá de los bounds originales
    A_min_original, A_max_original = MODEL_CONFIG["bounds"]["A"]
    A_min_dynamic = max(A_min_original, flux_min * 0.1)  # Al menos 10% del mínimo
    A_max_dynamic = min(A_max_original, flux_max * 10.0)  # Hasta 10x el máximo
    
    # Asegurar que haya un rango razonable
    if A_max_dynamic <= A_min_dynamic:
        A_max_dynamic = A_min_dynamic * 100.0
        A_max_dynamic = min(A_max_dynamic, A_max_original)
    
    A_bounds = (A_min_dynamic, A_max_dynamic)
    
    # Crear bounds dinámicos (copiar los de config y ajustar t0 y A)
    dynamic_bounds = MODEL_CONFIG["bounds"].copy()
    dynamic_bounds["t0"] = t0_bounds
    dynamic_bounds["A"] = A_bounds
    
    # Asegurar que p0 esté dentro de los bounds dinámicos ANTES de inicializar walkers
    if p0[2] < t0_bounds[0] or p0[2] > t0_bounds[1]:
        # Si está fuera, ajustarlo al centro del rango de fase
        p0[2] = (phase_min + phase_max) / 2.0
        if verbose:
            print(f"  [ADVERTENCIA] t0_guess ajustado a {p0[2]:.1f} (dentro de bounds dinámicos)")
    
    # Asegurar que p0[0] (A) esté dentro de los bounds dinámicos
    if p0[0] < A_bounds[0] or p0[0] > A_bounds[1]:
        # Si está fuera, ajustarlo al centro del rango dinámico
        p0[0] = (A_bounds[0] + A_bounds[1]) / 2.0
        if verbose:
            print(f"  [ADVERTENCIA] A_guess ajustado a {p0[0]:.2e} (dentro de bounds dinámicos)")
    
    n_walkers = MCMC_CONFIG["n_walkers"]
    n_steps = MCMC_CONFIG["n_steps"]
    burn_in = MCMC_CONFIG["burn_in"]
    random_seed = MCMC_CONFIG.get("random_seed", None)
    
    # Fijar semilla aleatoria para reproducibilidad
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Inicializar walkers de manera más robusta
    # Estrategia: distribuir walkers uniformemente en el espacio de parámetros
    # alrededor de p0, pero asegurando que estén dentro de bounds
    ndim = len(p0)
    param_names = MODEL_CONFIG["param_names"]
    
    # Calcular el tamaño típico de cada parámetro (rango del bound)
    param_scales = []
    for j, param_name in enumerate(param_names):
        bound_min, bound_max = dynamic_bounds[param_name]
        param_range = bound_max - bound_min
        p0_val = p0[j]
        # Escala: 10% del rango o 10% del valor absoluto de p0 (si p0 no es muy pequeño)
        scale_from_range = param_range * 0.1
        scale_from_p0 = abs(p0_val) * 0.1 if abs(p0_val) > 1e-10 else scale_from_range
        param_scales.append(max(scale_from_range, scale_from_p0))
    
    param_scales = np.array(param_scales)
    
    # Inicializar walkers con mejor estrategia para asegurar independencia lineal
    # Estrategia híbrida: algunos cerca de p0, otros distribuidos uniformemente
    pos = np.zeros((n_walkers, ndim))
    
    # Primer walker en p0
    pos[0] = p0.copy()
    
    # Para los demás walkers, usar una mezcla de estrategias
    for i in range(1, n_walkers):
        for j, param_name in enumerate(param_names):
            bound_min, bound_max = dynamic_bounds[param_name]
            margin = max((bound_max - bound_min) * 0.01, 1e-10)
            bound_min_safe = bound_min + margin
            bound_max_safe = bound_max - margin
            
            # Estrategia: alternar entre cerca de p0 y uniforme
            # Esto asegura diversidad en la inicialización
            if i % 2 == 1:
                # Walkers impares: cerca de p0 con dispersión
                pos[i, j] = p0[j] + param_scales[j] * np.random.randn()
            else:
                # Walkers pares: distribuidos uniformemente en el bound
                pos[i, j] = bound_min_safe + (bound_max_safe - bound_min_safe) * np.random.rand()
            
            # Asegurar que esté dentro de bounds
            pos[i, j] = np.clip(pos[i, j], bound_min_safe, bound_max_safe)
    
    # Verificar que los walkers no estén todos en el mismo punto
    # Si la desviación estándar de cualquier parámetro es muy pequeña, agregar más dispersión
    for j in range(ndim):
        std_j = np.std(pos[:, j])
        if std_j < 1e-10:
            # Si todos los walkers tienen el mismo valor, dispersarlos
            bound_min, bound_max = dynamic_bounds[param_names[j]]
            margin = max((bound_max - bound_min) * 0.01, 1e-10)
            bound_min_safe = bound_min + margin
            bound_max_safe = bound_max - margin
            # Dispersar uniformemente (excepto el primero que queda en p0)
            for i in range(1, n_walkers):
                pos[i, j] = bound_min_safe + (bound_max_safe - bound_min_safe) * np.random.rand()
    
    # Crear sampler con bounds dinámicos
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(phase, flux, flux_err, dynamic_bounds),
        threads=MCMC_CONFIG["n_threads"]
    )
    
    if verbose:
        print(f"  Bounds dinámicos para t0: ({t0_bounds[0]:.1f}, {t0_bounds[1]:.1f})")
        print(f"  Bounds dinámicos para A: ({A_bounds[0]:.2e}, {A_bounds[1]:.2e})")
        print(f"  Flujo observado: min={flux_min:.2e}, max={flux_max:.2e}, mediana={flux_median:.2e}")
        print(f"  Rango de fase observado: ({phase_min:.1f}, {phase_max:.1f}) días")
        print(f"Ejecutando MCMC con {n_walkers} walkers, {n_steps} pasos...")
    
    # Ejecutar MCMC
    sampler.run_mcmc(pos, n_steps, progress=verbose)
    
    # Obtener samples (después de burn-in)
    samples = sampler.chain[:, burn_in:, :].reshape(-1, ndim)
    
    # Calcular estadísticas
    # IMPORTANTE: param_medians calcula la mediana de CADA parámetro por separado
    # Esto significa: mediana_A, mediana_f, mediana_t0, etc.
    # Luego se evalúa el modelo con estos parámetros medianos: alerce_model(phase, mediana_A, mediana_f, ...)
    # NOTA: Esto NO es lo mismo que calcular la mediana de las curvas completas.
    # Si hay correlaciones entre parámetros, la combinación de medianas puede no estar
    # en el centro de las curvas rojas (que son samples individuales de la distribución).
    param_medians = np.median(samples, axis=0)
    param_std = np.std(samples, axis=0)
    param_percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    
    # Calcular modelo con parámetros medianos
    # Este es el modelo que se muestra como "MCMC Median" en los gráficos
    # y los valores que aparecen en "MCMC Fit Results"
    model_flux = alerce_model(phase, *param_medians)
    
    return {
        'params': param_medians,
        'params_err': param_std,
        'params_percentiles': param_percentiles,
        'samples': samples,
        'sampler': sampler,
        'model_flux': model_flux,
        'n_steps': n_steps,
        'burn_in': burn_in
    }

