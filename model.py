"""
Modelo de Villar (ALERCE) para curvas de luz de supernovas
"""
import numpy as np

def alerce_model(times, A, f, t0, t_rise, t_fall, gamma):
    """
    Modelo de Villar modificado por ALERCE
    
    Parameters:
    -----------
    times : array
        Tiempos (fase relativa)
    A, f, t0, t_rise, t_fall, gamma : float
        Parámetros del modelo
        
    Returns:
    --------
    flux : array
        Flujo modelado
    """
    t1 = t0 + gamma
    
    sigmoid = 1.0 / (1.0 + np.exp(-1/3 * (times - t1)))
    den = 1 + np.exp(-(times - t0) / t_rise)
    
    flux = (A * (1 - f) * np.exp(-(times - t1) / t_fall) / den * sigmoid
            + A * (1. - f * (times - t0) / gamma) / den * (1 - sigmoid))
    
    return flux

def flux_to_mag(flux):
    """Convertir flujo a magnitud"""
    return -2.5 * np.log10(flux)

def get_initial_guess(phase, flux):
    """
    Obtener estimación inicial de parámetros basada en los datos
    
    Returns:
    --------
    p0 : array [A, f, t0, t_rise, t_fall, gamma]
    """
    # Usar estadísticas más robustas
    max_flux = np.max(flux)
    median_flux = np.median(flux)
    flux_std = np.std(flux)
    
    # Encontrar el punto de máximo flujo
    max_idx = np.argmax(flux)
    t0_guess = phase[max_idx]
    
    # A_guess: usar el máximo observado con un factor conservador
    # Si hay mucha variación, usar percentil 90 en lugar del máximo absoluto
    if flux_std / (median_flux + 1e-10) > 0.5:  # Si hay mucha variación
        flux_90 = np.percentile(flux, 90)
        A_guess = flux_90 * 1.5
    else:
        # Si es relativamente plano, usar el máximo
        A_guess = max_flux * 1.2
    
    # Asegurar que A_guess sea razonable (no muy pequeño)
    A_guess = max(A_guess, max_flux * 1.1)
    
    f_guess = 0.5
    t_rise_guess = max(1.0, min(50.0, abs(t0_guess) / 2.0))  # Limitar t_rise
    t_fall_guess = 40.0
    gamma_guess = max(1.0, min(50.0, (phase.max() - phase.min()) / 2.0))  # Limitar gamma
    
    return np.array([A_guess, f_guess, t0_guess, t_rise_guess, t_fall_guess, gamma_guess])

