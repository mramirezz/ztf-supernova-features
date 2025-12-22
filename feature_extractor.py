"""
Extracción de features del modelo de Villar
"""
import numpy as np
from typing import Dict

def calculate_fit_statistics(phase, flux, flux_err, flux_model):
    """
    Calcular estadísticas de bondad de ajuste en el espacio de flujo
    
    Parameters:
    -----------
    phase : array
        Fases (días) - solo los puntos observados
    flux : array
        Flujo observado - solo los puntos observados
    flux_err : array
        Error en flujo observado - solo los puntos observados
    flux_model : array
        Flujo del modelo evaluado en los mismos puntos observados (phase)
    
    Returns:
    --------
    dict con: rms, mad, n_points, time_span
    """
    # Verificar que todos los arrays tengan la misma longitud
    assert len(flux) == len(flux_model), f"flux ({len(flux)}) y flux_model ({len(flux_model)}) deben tener la misma longitud"
    assert len(flux) == len(flux_err), f"flux ({len(flux)}) y flux_err ({len(flux_err)}) deben tener la misma longitud"
    assert len(flux) == len(phase), f"flux ({len(flux)}) y phase ({len(phase)}) deben tener la misma longitud"
    
    # Calcular residuales
    residuals = flux - flux_model
    
    # Verificar que no haya valores NaN o Inf
    valid_mask = np.isfinite(residuals) & np.isfinite(flux) & np.isfinite(flux_model) & np.isfinite(flux_err)
    valid_mask &= (flux_err > 0)  # Errores deben ser positivos
    
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"  [ADVERTENCIA] {n_invalid} puntos inválidos (NaN/Inf) serán excluidos del cálculo de métricas")
    
    residuals_valid = residuals[valid_mask]
    flux_valid = flux[valid_mask]
    flux_model_valid = flux_model[valid_mask]
    flux_err_valid = flux_err[valid_mask]
    
    n_points = len(residuals_valid)
    n_params = 6
    dof = n_points - n_params
    
    if n_points < n_params:
        return {
            'rms': np.inf,
            'mad': np.inf,
            'median_relative_error_pct': np.inf,
            'n_points': n_points,
            'time_span': phase.max() - phase.min() if len(phase) > 0 else 0
        }
    
    # RMS: raíz cuadrada de la media de los residuales al cuadrado
    rms = np.sqrt(np.sum(residuals_valid**2) / dof) if dof > 0 else np.inf
    
    # MAD: mediana de los valores absolutos de los residuales
    mad = np.median(np.abs(residuals_valid)) if len(residuals_valid) > 0 else np.inf
    
    # Error relativo porcentual: mediana del error relativo absoluto
    # Calculado como |(F_obs - F_model) / F_obs| * 100
    # Esto da el porcentaje de error relativo al flujo observado, sin usar errores observacionales
    # Evitar división por cero: solo usar puntos donde flux_valid > 0
    flux_positive_mask = flux_valid > 1e-10  # Evitar valores muy pequeños o cero
    if np.any(flux_positive_mask):
        relative_errors = np.abs(residuals_valid[flux_positive_mask] / flux_valid[flux_positive_mask]) * 100  # Porcentaje
        median_relative_error = np.median(relative_errors) if len(relative_errors) > 0 else np.inf
    else:
        median_relative_error = np.inf
    
    # NOTA: No calculamos chi-cuadrado reducido porque la propagación de errores
    # de magnitud a flujo hace que los errores absolutos en flujo sean muy pequeños
    # para objetos débiles, distorsionando el chi-cuadrado. RMS y MAD son métricas
    # más robustas que no dependen de los errores observacionales.
    
    time_span = phase.max() - phase.min() if len(phase) > 0 else 0
    
    return {
        'rms': rms,
        'mad': mad,
        'median_relative_error_pct': median_relative_error,
        'n_points': n_points,
        'time_span': time_span
    }

def extract_features(mcmc_results, phase, flux, flux_err, sn_name, filter_name):
    """
    Extraer las 25 features del modelo de Villar
    
    Parameters:
    -----------
    mcmc_results : dict
        Resultados del ajuste MCMC (debe incluir 'model_flux')
    phase, flux, flux_err : arrays
        Datos observados en flujo
    sn_name, filter_name : str
        Identificadores
        
    Returns:
    --------
    dict con todas las features
    """
    params = mcmc_results['params']
    params_err = mcmc_results['params_err']
    flux_model = mcmc_results['model_flux']
    
    # Parámetros principales
    features = {
        'sn_name': sn_name,
        'filter_band': filter_name,
        'A': params[0],
        'f': params[1],
        't0': params[2],
        't_rise': params[3],
        't_fall': params[4],
        'gamma': params[5],
    }
    
    # Errores formales (usamos std de MCMC)
    features.update({
        'A_err': params_err[0],
        'f_err': params_err[1],
        't0_err': params_err[2],
        't_rise_err': params_err[3],
        't_fall_err': params_err[4],
        'gamma_err': params_err[5],
    })
    
    # Errores Monte Carlo (mismo que formales en MCMC, pero mantenemos nombre)
    features.update({
        'A_mc_std': params_err[0],
        'f_mc_std': params_err[1],
        't0_mc_std': params_err[2],
        't_rise_mc_std': params_err[3],
        't_fall_mc_std': params_err[4],
        'gamma_mc_std': params_err[5],
    })
    
    # Estadísticas de ajuste (calculadas en flujo)
    stats = calculate_fit_statistics(phase, flux, flux_err, flux_model)
    features.update({
        'rms': stats['rms'],
        'mad': stats['mad'],
        'median_relative_error_pct': stats['median_relative_error_pct'],
        'n_points': stats['n_points'],
        'time_span': stats['time_span']
    })
    
    return features

