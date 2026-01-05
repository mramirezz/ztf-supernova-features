"""
Ajuste con MCMC usando emcee
"""
import numpy as np
import emcee
from typing import Dict, Tuple
from scipy.stats import norm
from model import alerce_model
from config import MCMC_CONFIG, MODEL_CONFIG

def log_likelihood(params, times, flux, flux_err, dynamic_bounds=None, is_upper_limit=None):
    """
    Log-likelihood para MCMC
    
    Parameters:
    -----------
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos son upper limits.
        Para upper limits, el modelo debe estar por debajo del límite.
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
        
        # Separar datos normales y upper limits
        if is_upper_limit is not None and np.any(is_upper_limit):
            # Datos normales: chi-cuadrado estándar
            mask_normal = ~is_upper_limit
            chi2_normal = 0.0
            if np.any(mask_normal):
                chi2_normal = np.sum(((flux[mask_normal] - model_flux[mask_normal]) / flux_err[mask_normal]) ** 2)
            
            # Upper limits: usar método CDF para datos censurados (estadísticamente riguroso)
            # Basado en Ivezić et al. (2014), Capítulo 4.2.7
            mask_ul = is_upper_limit
            log_L_ul = 0.0
            if np.any(mask_ul):
                flux_ul = flux[mask_ul]  # Valores de los upper limits
                flux_model_ul = model_flux[mask_ul]  # Modelo evaluado en tiempos de upper limits
                
                # Estimar σ_i para cada upper limit
                # Si tenemos flux_err para upper limits, usarlo; sino estimar
                flux_err_ul = flux_err[mask_ul]
                
                # Si flux_err es NaN o no disponible, estimar basado en el límite
                # Usar 5% del límite (más estricto que 10%) para hacer la penalización más fuerte
                # cuando el modelo excede el límite. Esto hace que z sea más negativo cuando
                # flux_model > flux_ul, resultando en una penalización más severa.
                sigma_ul = np.where(
                    np.isfinite(flux_err_ul) & (flux_err_ul > 0),
                    flux_err_ul,
                    flux_ul * 0.05  # 5% del límite como fallback (más estricto que 10%)
                )
                
                # Calcular z = (flux_ul - flux_model) / σ
                # Si el modelo está por debajo del límite: z > 0 → CDF alta → log_L alto (bueno)
                # Si el modelo excede el límite: z < 0 → CDF baja → log_L bajo (malo)
                z_ul = (flux_ul - flux_model_ul) / (sigma_ul + 1e-10)  # +1e-10 para evitar división por cero
                
                # Calcular CDF: P(flux ≤ flux_ul | modelo) = Φ(z)
                # Permitir valores más negativos de z para penalizaciones más fuertes
                # Clip a -20 en lugar de -10 para permitir penalizaciones más extremas
                z_ul_clipped = np.clip(z_ul, -20, 10)  # CDF es ~0 para z < -20, ~1 para z > 10
                cdf_ul = norm.cdf(z_ul_clipped)
                
                # Proteger contra valores de CDF muy pequeños que causarían log → -∞
                # Usar un mínimo más pequeño (equivalente a ~8σ) para permitir penalizaciones más fuertes
                # Esto permite que cuando el modelo excede mucho el límite, la penalización sea más severa
                cdf_ul = np.maximum(cdf_ul, 1e-15)
                
                # Log-likelihood para upper limits: ln[P(flux ≤ flux_ul | modelo)]
                log_L_ul = np.sum(np.log(cdf_ul))
            
            return -0.5 * chi2_normal + log_L_ul
        else:
            # Sin upper limits: chi-cuadrado estándar
            chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
            return -0.5 * chi2
    except:
        return -np.inf

def log_prior(params, dynamic_bounds=None, times=None, flux=None, is_upper_limit=None):
    """
    Prior uniforme dentro de los bounds, con restricción adicional para upper limits
    
    Si hay upper limits, rechaza explícitamente parámetros que hacen que el modelo
    exceda cualquier upper limit. Esto es más estricto que solo la penalización en el likelihood.
    """
    A, f, t0, t_rise, t_fall, gamma = params
    
    # Usar bounds dinámicos si se proporcionan, sino usar los de config
    if dynamic_bounds is None:
        bounds = MODEL_CONFIG["bounds"]
    else:
        bounds = dynamic_bounds
    
    # Verificar bounds básicos
    if not (bounds["A"][0] < A < bounds["A"][1] and
            bounds["f"][0] < f < bounds["f"][1] and
            bounds["t0"][0] < t0 < bounds["t0"][1] and
            bounds["t_rise"][0] < t_rise < bounds["t_rise"][1] and
            bounds["t_fall"][0] < t_fall < bounds["t_fall"][1] and
            bounds["gamma"][0] < gamma < bounds["gamma"][1]):
        return -np.inf
    
    # Restricción adicional: penalizar fuertemente parámetros que exceden upper limits
    # Usamos una penalización muy fuerte pero finita en lugar de -np.inf para evitar
    # problemas numéricos en emcee cuando muchos walkers tienen log_prob = -np.inf
    if is_upper_limit is not None and np.any(is_upper_limit) and times is not None and flux is not None:
        try:
            model_flux = alerce_model(times, A, f, t0, t_rise, t_fall, gamma)
            
            # Verificar que el modelo sea válido antes de verificar upper limits
            if np.any(np.isnan(model_flux)) or np.any(np.isinf(model_flux)) or np.any(model_flux <= 0):
                return -np.inf
            
            flux_ul = flux[is_upper_limit]
            flux_model_ul = model_flux[is_upper_limit]
            
            # Verificar que los valores del modelo en upper limits sean válidos
            if np.any(np.isnan(flux_model_ul)) or np.any(np.isinf(flux_model_ul)):
                return -np.inf
            
            # Penalización fuerte (pero finita) si el modelo excede upper limits
            # Esto es más robusto numéricamente que -np.inf
            # La penalización es relativa al exceso normalizado, escalada por un factor grande
            # pero más generoso que antes
            excess = flux_model_ul - flux_ul
            if np.any(excess > 0):
                # Penalización cuadrática fuerte pero más generosa: -1e8 * suma((exceso/flux_ul)²)
                # Reducido de -1e10 a -1e8 para ser más generoso
                relative_excess = excess[excess > 0] / flux_ul[excess > 0]
                penalty = -1e8 * np.sum(relative_excess ** 2)
                return penalty
        except:
            # Si hay error al evaluar el modelo, rechazar
            return -np.inf
    
    return 0.0

def log_posterior(params, times, flux, flux_err, dynamic_bounds=None, is_upper_limit=None):
    """
    Log posterior = log prior + log likelihood
    """
    # Pasar información de upper limits al prior para restricción explícita
    lp = log_prior(params, dynamic_bounds, times=times, flux=flux, is_upper_limit=is_upper_limit)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, times, flux, flux_err, dynamic_bounds, is_upper_limit)

def fit_mcmc(phase, flux, flux_err, p0=None, verbose=True, is_upper_limit=None):
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
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos son upper limits
        
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
    t0_margin = max(100.0, phase_range * 0.5)  # 50% del rango o mínimo 100 días (más generoso)
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
    A_min_dynamic = max(A_min_original, flux_min * 0.01)  # Al menos 1% del mínimo (más generoso)
    A_max_dynamic = min(A_max_original, flux_max * 50.0)  # Hasta 50x el máximo (más generoso)
    
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
        args=(phase, flux, flux_err, dynamic_bounds, is_upper_limit),
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
    
    # Filtrar samples que exceden upper limits (si hay upper limits)
    # Aunque el MCMC penaliza estos samples, la mediana/promedio de parámetros individuales
    # puede no respetar las restricciones debido a correlaciones entre parámetros.
    # Filtrar asegura que solo usamos samples físicamente válidos para calcular estadísticas.
    if is_upper_limit is not None and np.any(is_upper_limit):
        valid_samples = []
        flux_ul = flux[is_upper_limit]
        ul_times = phase[is_upper_limit]
        
        for sample in samples:
            try:
                model_flux_sample = alerce_model(phase, *sample)
                # Verificar que el modelo no exceda ningún upper limit
                flux_model_ul = model_flux_sample[is_upper_limit]
                # El modelo debe estar por debajo de todos los upper limits (con pequeña tolerancia numérica)
                if np.all(flux_model_ul <= flux_ul * 1.01):  # 1% de tolerancia numérica
                    valid_samples.append(sample)
            except:
                continue
        
        if len(valid_samples) > 0:
            samples_valid = np.array(valid_samples)
            # Si tenemos suficientes samples válidos (al menos 20% del total), usarlos
            # Si no, usar todos (mejor que nada, pero con advertencia)
            if len(samples_valid) < len(samples) * 0.2:
                if verbose:
                    print(f"  [ADVERTENCIA] Solo {len(samples_valid)}/{len(samples)} samples respetan upper limits ({100*len(samples_valid)/len(samples):.1f}%)")
                    print(f"  [ADVERTENCIA] Usando todos los samples para estadísticas (puede que la mediana exceda upper limits)")
                samples_valid = samples
        else:
            # Si ningún sample es válido, usar todos (mejor que nada)
            if verbose:
                print(f"  [ADVERTENCIA] Ningún sample respeta upper limits, usando todos los samples")
            samples_valid = samples
    else:
        samples_valid = samples
    
    # Calcular estadísticas usando solo samples válidos
    # IMPORTANTE: param_medians calcula la mediana de CADA parámetro por separado
    # Esto significa: mediana_A, mediana_f, mediana_t0, etc.
    # Luego se evalúa el modelo con estos parámetros medianos: alerce_model(phase, mediana_A, mediana_f, ...)
    # NOTA: Esto NO es lo mismo que calcular la mediana de las curvas completas.
    # Si hay correlaciones entre parámetros, la combinación de medianas puede no estar
    # en el centro de las curvas rojas (que son samples individuales de la distribución).
    param_medians = np.median(samples_valid, axis=0)
    param_std = np.std(samples_valid, axis=0)
    param_percentiles = np.percentile(samples_valid, [16, 50, 84], axis=0)
    
    # Calcular modelo con parámetros medianos
    # Este es el modelo que se muestra como "MCMC Median" en los gráficos
    # y los valores que aparecen en "MCMC Fit Results"
    model_flux = alerce_model(phase, *param_medians)
    
    # ==========================================================================
    # CALCULAR MEDIAN OF CURVES (curva central)
    # ==========================================================================
    # Seleccionar las 500 curvas con MEJOR LOG-LIKELIHOOD de 2000 candidatas
    # Luego encontrar la curva más cercana a la mediana de esas 500
    # Esto captura mejor las correlaciones y da curvas que ajustan bien los datos
    
    # PASO 1: Evaluar 2000 candidatas y calcular su log-likelihood
    n_candidates = min(2000, len(samples_valid))
    step_candidates = max(1, len(samples_valid) // n_candidates)
    candidate_indices = np.arange(0, len(samples_valid), step_candidates)[:n_candidates]
    
    # Usar flux_err si está disponible
    if flux_err is not None and len(flux_err) == len(flux):
        sigma = flux_err.copy()
        sigma = np.where(sigma > 0, sigma, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1e-10)
    else:
        sigma = np.ones_like(flux) * np.std(flux) * 0.1
    
    candidate_loglik = []
    valid_candidate_indices = []
    for idx in candidate_indices:
        try:
            flux_model_candidate = alerce_model(phase, *samples_valid[idx])
            flux_model_candidate = np.clip(flux_model_candidate, 1e-10, None)
            if np.all(np.isfinite(flux_model_candidate)) and np.all(flux_model_candidate > 0):
                chi2 = np.sum(((flux - flux_model_candidate) / sigma)**2)
                log_lik = -0.5 * chi2
                candidate_loglik.append(log_lik)
                valid_candidate_indices.append(idx)
        except:
            continue
    
    # PASO 2: Seleccionar las 500 con mayor log-likelihood
    params_median_of_curves = None
    central_curve_idx = None
    if len(candidate_loglik) >= 10:
        n_samples_for_moc = min(500, len(valid_candidate_indices))
        sorted_idx = np.argsort(candidate_loglik)[::-1][:n_samples_for_moc]
        moc_indices = np.array(valid_candidate_indices)[sorted_idx]
        samples_for_moc = samples_valid[moc_indices]
        
        # PASO 3: Evaluar las curvas seleccionadas y encontrar la más central
        all_flux_curves = []
        valid_sample_indices = []
        for i, sample in enumerate(samples_for_moc):
            try:
                flux_curve = alerce_model(phase, *sample)
                flux_curve = np.clip(flux_curve, 1e-10, None)
                if np.all(np.isfinite(flux_curve)) and np.all(flux_curve > 0) and np.all(flux_curve < 1e10):
                    all_flux_curves.append(flux_curve)
                    valid_sample_indices.append(i)
            except:
                continue
        
        if len(all_flux_curves) >= 10:
            all_flux_curves = np.array(all_flux_curves)
            flux_p50 = np.percentile(all_flux_curves, 50, axis=0)
            distances = np.sum((all_flux_curves - flux_p50)**2, axis=1)
            best_idx_local = np.argmin(distances)
            original_idx = valid_sample_indices[best_idx_local]
            params_median_of_curves = samples_for_moc[original_idx]
            central_curve_idx = moc_indices[original_idx]
    
    return {
        'params': param_medians,
        'params_err': param_std,
        'params_percentiles': param_percentiles,
        'params_median_of_curves': params_median_of_curves,  # Parámetros de la curva central
        'central_curve_idx': central_curve_idx,  # Índice del sample de la curva central
        'samples': samples,  # Todos los samples (para corner plot y visualización completa)
        'samples_valid': samples_valid,  # Solo samples que respetan upper limits (para estadísticas)
        'sampler': sampler,
        'model_flux': model_flux,
        'n_steps': n_steps,
        'burn_in': burn_in
    }

def validate_physical_fit(mcmc_results, phase, flux, is_upper_limit=None):
    """
    Validar que el fit MCMC tenga comportamiento físico razonable
    
    Validaciones:
    1. El flujo antes de la primera detección debe ser menor que en la primera detección
    2. El flujo después de la última detección debe ser menor que en la última detección
    
    Parameters:
    -----------
    mcmc_results : dict
        Resultados del MCMC (debe contener 'params' con parámetros medianos)
    phase : array
        Fases (tiempos relativos) de las observaciones
    flux : array
        Flujos observados (para identificar primera y última detección)
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos son upper limits
        
    Returns:
    --------
    is_valid : bool
        True si el fit es físicamente válido, False en caso contrario
    reason : str
        Razón del rechazo si is_valid=False, None si es válido
    """
    from model import alerce_model
    
    # Obtener parámetros medianos
    params = mcmc_results['params']
    
    # Identificar primera y última detección normal (excluyendo upper limits)
    if is_upper_limit is not None and np.any(is_upper_limit):
        mask_normal = ~is_upper_limit
        phase_normal = phase[mask_normal]
        flux_normal = flux[mask_normal]
    else:
        phase_normal = phase
        flux_normal = flux
    
    if len(phase_normal) == 0:
        return False, "No hay detecciones normales"
    
    # Primera y última detección normal
    first_idx = np.argmin(phase_normal)
    last_idx = np.argmax(phase_normal)
    
    first_phase = phase_normal[first_idx]
    first_flux = flux_normal[first_idx]
    
    last_phase = phase_normal[last_idx]
    last_flux = flux_normal[last_idx]
    
    # Validación 1: Flujo antes de la primera detección debe ser menor
    t_early = first_phase - 500.0  # 500 días antes de la primera detección
    try:
        flux_early = alerce_model(np.array([t_early]), *params)[0]
        if flux_early > first_flux:
            return False, f"Flujo no físico antes de primera detección: {flux_early:.2e} > {first_flux:.2e} en t={t_early:.1f}"
    except:
        return False, "Error al evaluar modelo en tiempo temprano"
    
    # Validación 2: Flujo después de la última detección debe ser menor
    t_late = last_phase + 500.0  # 500 días después de la última detección
    try:
        flux_late = alerce_model(np.array([t_late]), *params)[0]
        if flux_late > last_flux:
            return False, f"Flujo no físico después de última detección: {flux_late:.2e} > {last_flux:.2e} en t={t_late:.1f}"
    except:
        return False, "Error al evaluar modelo en tiempo tardío"
    
    return True, None

