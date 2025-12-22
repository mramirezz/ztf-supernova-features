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
        Fases (días)
    flux : array
        Flujo observado
    flux_err : array
        Error en flujo observado
    flux_model : array
        Flujo del modelo
    
    Returns:
    --------
    dict con: rms, mad, reduced_chi2, n_points, time_span
    """
    residuals = flux - flux_model
    n_points = len(flux)
    n_params = 6
    dof = n_points - n_params
    
    rms = np.sqrt(np.sum(residuals**2) / dof) if dof > 0 else np.inf
    mad = np.median(np.abs(residuals))
    
    chi2 = np.sum((residuals / flux_err)**2)
    reduced_chi2 = chi2 / dof if dof > 0 else np.inf
    
    time_span = phase.max() - phase.min()
    
    return {
        'rms': rms,
        'mad': mad,
        'reduced_chi2': reduced_chi2,
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
        'reduced_chi2': stats['reduced_chi2'],
        'n_points': stats['n_points'],
        'time_span': stats['time_span']
    })
    
    return features

