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
        # Diagnosticar qué tipo de valores inválidos hay
        n_nan_residuals = np.sum(~np.isfinite(residuals))
        n_nan_flux = np.sum(~np.isfinite(flux))
        n_nan_flux_model = np.sum(~np.isfinite(flux_model))
        n_nan_flux_err = np.sum(~np.isfinite(flux_err))
        n_zero_flux_err = np.sum((flux_err <= 0) & np.isfinite(flux_err))
        
        # Solo mostrar warning si hay valores inválidos significativos
        if n_invalid > 0:
            details = []
            if n_nan_residuals > 0:
                details.append(f"{n_nan_residuals} residuales inválidos")
            if n_nan_flux > 0:
                details.append(f"{n_nan_flux} flujos observados inválidos")
            if n_nan_flux_model > 0:
                details.append(f"{n_nan_flux_model} flujos del modelo inválidos")
            if n_nan_flux_err > 0:
                details.append(f"{n_nan_flux_err} errores de flujo inválidos")
            if n_zero_flux_err > 0:
                details.append(f"{n_zero_flux_err} errores de flujo <= 0")
            
            detail_str = ", ".join(details) if details else "valores inválidos"
            print(f"  [ADVERTENCIA] {n_invalid} puntos inválidos ({detail_str}) serán excluidos del cálculo de métricas")
    
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
    Extraer las 25+ features del modelo de Villar
    
    Parameters:
    -----------
    mcmc_results : dict
        Resultados del ajuste MCMC (debe incluir 'model_flux' y 'params_median_of_curves')
    phase, flux, flux_err : arrays
        Datos observados en flujo
    sn_name, filter_name : str
        Identificadores
        
    Returns:
    --------
    dict con todas las features, incluyendo:
        - Parámetros de Median of Params (sin sufijo): A, f, t0, etc.
        - Parámetros de Median of Curves (sufijo _moc): A_moc, f_moc, t0_moc, etc.
    """
    params = mcmc_results['params']  # Median of Params
    params_err = mcmc_results['params_err']
    flux_model = mcmc_results['model_flux']
    params_moc = mcmc_results.get('params_median_of_curves', None)  # Median of Curves
    
    # Parámetros principales (Median of Params - línea azul punteada)
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
    
    # Parámetros de Median of Curves (línea verde sólida)
    # Estos capturan mejor las correlaciones entre parámetros
    if params_moc is not None:
        features.update({
            'A_moc': params_moc[0],
            'f_moc': params_moc[1],
            't0_moc': params_moc[2],
            't_rise_moc': params_moc[3],
            't_fall_moc': params_moc[4],
            'gamma_moc': params_moc[5],
        })
    else:
        # Si no hay curva central, usar NaN
        features.update({
            'A_moc': np.nan,
            'f_moc': np.nan,
            't0_moc': np.nan,
            't_rise_moc': np.nan,
            't_fall_moc': np.nan,
            'gamma_moc': np.nan,
        })
    
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

