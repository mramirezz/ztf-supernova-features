"""
Generación de gráficos para visualización de ajustes
"""
import matplotlib.pyplot as plt
import matplotlib

# =============================================================================
# ESTILO A&A (Astronomy & Astrophysics) - Computer Modern / LaTeX style
# =============================================================================

# Usar estilo LaTeX (Computer Modern fonts)
matplotlib.rcParams['text.usetex'] = False  # True requiere LaTeX instalado
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern para math
matplotlib.rcParams['axes.unicode_minus'] = False  # Usar guión ASCII para signo menos

# Tamaños de fuente estilo A&A (compactos)
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['legend.fontsize'] = 6
matplotlib.rcParams['figure.titlesize'] = 10

# Líneas finas estilo journal
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['grid.linewidth'] = 0.3
matplotlib.rcParams['grid.alpha'] = 0.4
matplotlib.rcParams['lines.linewidth'] = 1.2
matplotlib.rcParams['lines.markersize'] = 4

# Spines - todos visibles para A&A
matplotlib.rcParams['axes.spines.top'] = True
matplotlib.rcParams['axes.spines.right'] = True

# Leyenda muy compacta
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.framealpha'] = 0.95
matplotlib.rcParams['legend.edgecolor'] = '0.7'
matplotlib.rcParams['legend.borderpad'] = 0.2
matplotlib.rcParams['legend.handlelength'] = 1.0
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['legend.labelspacing'] = 0.2

# Ticks hacia adentro (estilo A&A)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

# Padding mínimo
matplotlib.rcParams['axes.titlepad'] = 3
matplotlib.rcParams['axes.labelpad'] = 2

import numpy as np
from typing import Dict
from config import PLOT_CONFIG

def plot_fit(phase, mag, mag_err, mag_model, flux, flux_model, 
              sn_name, filter_name, save_path=None):
    """
    Generar gráfico de ajuste (magnitud y flujo)
    
    Parameters:
    -----------
    phase, mag, mag_err : arrays
        Datos observados
    mag_model, flux_model : arrays
        Modelo ajustado
    sn_name, filter_name : str
        Identificadores
    save_path : str, optional
        Ruta para guardar figura
    """
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["figsize"])
    
    # Magnitude plot
    axes[0].errorbar(phase, mag, yerr=mag_err, fmt='o', alpha=0.6, 
                     label='Observations', markersize=4)
    axes[0].plot(phase, mag_model, 'r-', linewidth=2, label='MCMC Model')
    axes[0].set_xlabel('Phase (days)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'{sn_name} - Filter {filter_name} (Magnitude)')
    axes[0].legend()
    axes[0].grid(False)  # A&A style: no background grid
    axes[0].invert_yaxis()
    
    # Flux plot
    axes[1].errorbar(phase, flux, yerr=None, fmt='o', alpha=0.6,
                     label='Observations', markersize=4)
    axes[1].plot(phase, flux_model, 'r-', linewidth=2, label='MCMC Model')
    axes[1].set_xlabel('Phase (days)')
    axes[1].set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$)')
    axes[1].set_title(f'{sn_name} - Filter {filter_name} (Flux)')
    axes[1].legend()
    axes[1].grid(False)  # A&A style: no background grid
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
        plt.close(fig)  # Liberar memoria de la figura
        return None
    else:
        return fig

def plot_fit_with_uncertainty(phase, mag, mag_err, mag_model, flux, flux_model,
                              samples, n_samples_to_show=50,
                              sn_name=None, filter_name=None, save_path=None,
                              phase_ul=None, mag_ul=None, flux_ul=None,
                              is_upper_limit=None, flux_err=None, had_upper_limits=None,
                              xlim=None, param_medians_phase_relative=None,
                              param_medians=None, params_median_of_curves=None):
    """
    Generar gráfico de ajuste mostrando la mediana de parámetros con bandas de confianza
    
    Parameters:
    -----------
    phase, mag, mag_err : arrays
        Datos observados (puntos de observación + upper limits usados en fit)
    mag_model, flux_model : arrays
        Modelo ajustado (mediana) evaluado en los puntos observados
    samples : array (n_samples, n_params)
        TODOS los samples del MCMC (para calcular bandas de confianza)
    n_samples_to_show : int
        No usado (mantenido por compatibilidad)
    phase_ul, mag_ul, flux_ul : arrays, optional
        Upper limits para mostrar en el gráfico (NO usados en fit)
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos en phase/mag/flux son upper limits usados en el fit
    flux_err : array, optional
        Errores en flujo (si no se proporciona, se calcula desde mag_err o se omite)
    had_upper_limits : bool, optional
        Indica si había upper limits ANTES de combinarlos con los datos (para determinar extensión del plot)
    sn_name, filter_name : str
        Identificadores
    save_path : str, optional
        Ruta para guardar figura
    """
    from model import alerce_model, flux_to_mag
    
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["figsize"], sharex=True)
    fig.subplots_adjust(hspace=0.05, top=0.92, bottom=0.12, left=0.12, right=0.97)
    fig.suptitle(f'{sn_name} - Filter {filter_name}', fontsize=10, fontweight='bold', y=0.97)
    
    # Crear array de fase denso para curvas suaves
    phase_min = phase.min()
    phase_max = phase.max()
    
    # Usar had_upper_limits si está disponible, sino verificar is_upper_limit
    if had_upper_limits is not None:
        has_ul = had_upper_limits
    else:
        has_ul = (is_upper_limit is not None and np.any(is_upper_limit))
    
    # Extensión antes del mínimo: 5 días si había upper limits, 10 días si no
    extension_before = 5.0 if has_ul else 10.0
    extension_after = 10.0  # Siempre 10 días después para mostrar la caída
    
    phase_smooth = np.arange(phase_min - extension_before, phase_max + extension_after + 0.5, 1.0)
    
    # ==========================================================================
    # CALCULAR AMBAS MEDIANAS Y BANDAS DE CONFIANZA
    # ==========================================================================
    # 1. Mediana de parámetros: curva(median(A), median(f), ...) - línea punteada
    # 2. Mediana de curvas: percentil 50 de todas las curvas - línea sólida
    # Las bandas muestran los percentiles de las curvas de todos los samples válidos
    # ==========================================================================
    
    # Usar param_medians pasado como parámetro (viene de mcmc_results['params'])
    # Si no se pasa, calcular desde samples (fallback)
    if param_medians is None and samples is not None:
        param_medians = np.median(samples, axis=0)
    param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
    
    # Calcular curva de la mediana de parámetros
    flux_model_params = None  # Curva de mediana de parámetros
    mag_model_params = None
    
    if param_medians is not None:
        try:
            flux_model_params = alerce_model(phase_smooth, *param_medians)
            flux_model_params = np.clip(flux_model_params, 1e-10, None)
            mag_model_params = flux_to_mag(flux_model_params)
        except:
            pass
        
        # DEBUG: Mostrar parámetros de la mediana
        is_param_medians_mjd = param_medians[2] > 50000
        print(f"    [DEBUG] Parámetros MEDIANA - {'EN MJD' if is_param_medians_mjd else 'EN FASE RELATIVA'}:")
        for i, name in enumerate(param_names):
            print(f"      {name} = {param_medians[i]:.6e}")
        
        if param_medians_phase_relative is not None:
            print(f"    [DEBUG] Parámetros MEDIANA (fase relativa, referencia):")
            for i, name in enumerate(param_names):
                print(f"      {name} = {param_medians_phase_relative[i]:.6e}")
    
    # Inicializar mediana de curvas (se calculará abajo si hay suficientes samples)
    flux_model_curves = None  # Mediana de curvas (percentil 50)
    mag_model_curves = None
    
    # Calcular bandas de confianza (percentiles de las curvas de samples)
    flux_band_1sigma = None  # (lower_16, upper_84)
    flux_band_2sigma = None  # (lower_2.5, upper_97.5)
    mag_band_1sigma = None
    mag_band_2sigma = None
    sample_indices = None  # Inicializar para retornar después
    best_sample_global_idx = None  # Índice del sample de la curva central (para plot_extended_model)
    flux_curves_to_plot = None  # Curvas individuales para plotear como líneas rojas
    mag_curves_to_plot = None
    
    if samples is not None and len(samples) > 0:
        # PASO 1: Evaluar curvas candidatas y calcular su log-likelihood
        # Luego seleccionar las 500 con MAYOR likelihood (mejor ajuste)
        n_candidates = min(2000, len(samples))  # Evaluar hasta 2000 candidatas
        step = max(1, len(samples) // n_candidates)
        candidate_indices = np.arange(0, len(samples), step)[:n_candidates]
        
        # Calcular log-likelihood de cada curva (chi2 simplificado)
        candidate_loglik = []
        valid_indices = []
        
        # Usar flux_err si está disponible, sino asumir error constante
        if flux_err is not None and len(flux_err) == len(flux):
            sigma = flux_err
            # Reemplazar errores <= 0 con un valor pequeño
            sigma = np.where(sigma > 0, sigma, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1e-10)
        else:
            sigma = np.ones_like(flux) * np.std(flux) * 0.1  # 10% del std como error
        
        for idx in candidate_indices:
            try:
                # Evaluar en los puntos de datos
                flux_model = alerce_model(phase, *samples[idx])
                flux_model = np.clip(flux_model, 1e-10, None)
                
                if np.all(np.isfinite(flux_model)) and np.all(flux_model > 0):
                    # Log-likelihood gaussiano: -0.5 * sum((obs - model)^2 / sigma^2)
                    chi2 = np.sum(((flux - flux_model) / sigma)**2)
                    log_lik = -0.5 * chi2
                    candidate_loglik.append(log_lik)
                    valid_indices.append(idx)
            except:
                continue
        
        # PASO 2: Seleccionar las 500 con MAYOR log-likelihood
        n_samples_for_bands = min(500, len(valid_indices))
        if len(candidate_loglik) > 0:
            # Ordenar de mayor a menor log-likelihood
            sorted_idx = np.argsort(candidate_loglik)[::-1][:n_samples_for_bands]
            sample_indices = np.array(valid_indices)[sorted_idx]
            best_loglik = candidate_loglik[sorted_idx[0]]
            worst_loglik = candidate_loglik[sorted_idx[-1]] if len(sorted_idx) > 1 else best_loglik
            print(f"    [DEBUG] Seleccionadas {len(sample_indices)} curvas con mejor likelihood de {len(valid_indices)} candidatas")
            print(f"    [DEBUG] Log-likelihood: mejor={best_loglik:.1f}, peor seleccionada={worst_loglik:.1f}")
        else:
            sample_indices = np.array([])
        
        # PASO 3: Evaluar las curvas seleccionadas en phase_smooth para plotear
        all_flux_curves = []
        for idx in sample_indices:
            try:
                flux_curve = alerce_model(phase_smooth, *samples[idx])
                flux_curve = np.clip(flux_curve, 1e-10, None)
                if np.all(np.isfinite(flux_curve)) and np.all(flux_curve > 0) and np.all(flux_curve < 1e10):
                    all_flux_curves.append(flux_curve)
            except:
                continue
        
        if len(all_flux_curves) >= 10:  # Necesitamos suficientes curvas para percentiles
            all_flux_curves = np.array(all_flux_curves)
            
            # Calcular percentiles para bandas de confianza
            # 1-sigma: 68% del intervalo (percentiles 16-84)
            flux_p16 = np.percentile(all_flux_curves, 16, axis=0)
            flux_p50 = np.percentile(all_flux_curves, 50, axis=0)  # Mediana de curvas
            flux_p84 = np.percentile(all_flux_curves, 84, axis=0)
            flux_band_1sigma = (flux_p16, flux_p84)
            
            # 2-sigma: 95% del intervalo (percentiles 2.5-97.5)
            flux_p2_5 = np.percentile(all_flux_curves, 2.5, axis=0)
            flux_p97_5 = np.percentile(all_flux_curves, 97.5, axis=0)
            flux_band_2sigma = (flux_p2_5, flux_p97_5)
            
            # Convertir a magnitud
            mag_band_1sigma = (flux_to_mag(flux_p84), flux_to_mag(flux_p16))  # Invertido porque mag inversa
            mag_band_2sigma = (flux_to_mag(flux_p97_5), flux_to_mag(flux_p2_5))
            
            # Calcular curva central (Median of Curves) usando las MISMAS curvas que ploteamos
            # Esto garantiza que la curva verde esté dentro de las curvas rojas
            # NO usar params_median_of_curves precalculado porque puede ser de samples diferentes
            distances = np.sum((all_flux_curves - flux_p50)**2, axis=1)
            best_curve_idx_local = np.argmin(distances)  # Índice dentro de all_flux_curves
            flux_model_curves = all_flux_curves[best_curve_idx_local]
            mag_model_curves = flux_to_mag(flux_model_curves)
            
            # Guardar el índice GLOBAL del sample (para reutilizar en plot_extended_model)
            best_sample_global_idx = sample_indices[best_curve_idx_local]
            print(f"    [DEBUG] Curva central: índice local {best_curve_idx_local}, sample global {best_sample_global_idx}")
            
            # Convertir todas las curvas a magnitud para plotear líneas individuales
            # Plotear todas las 500 curvas
            flux_curves_to_plot = all_flux_curves
            mag_curves_to_plot = np.array([flux_to_mag(fc) for fc in flux_curves_to_plot])
            
            print(f"    [DEBUG] Bandas de confianza calculadas con {len(all_flux_curves)} curvas válidas")
            print(f"    [DEBUG] Ploteando {len(flux_curves_to_plot)} curvas individuales")
            
            # Liberar memoria del array grande (pero guardar las curvas para plotear)
            del all_flux_curves
    
    # ==========================================================================
    # SEPARAR OBSERVACIONES NORMALES DE UPPER LIMITS
    # ==========================================================================
    if is_upper_limit is not None and np.any(is_upper_limit):
        mask_normal = ~is_upper_limit
        mask_ul_fit = is_upper_limit
        
        phase_normal_mag = phase[mask_normal]
        mag_normal = mag[mask_normal]
        mag_err_normal = mag_err[mask_normal] if mag_err is not None and len(mag_err) == len(mag) else None
        
        phase_ul_fit_mag = phase[mask_ul_fit]
        mag_ul_fit = mag[mask_ul_fit]
    else:
        phase_normal_mag = phase
        mag_normal = mag
        mag_err_normal = mag_err if (mag_err is not None and len(mag_err) == len(mag)) else None
        phase_ul_fit_mag = np.array([])
        mag_ul_fit = np.array([])
    
    # ==========================================================================
    # PLOT DE MAGNITUD
    # ==========================================================================
    
    # Curvas individuales (líneas rojas semitransparentes) - plotear primero (zorder bajo)
    if mag_curves_to_plot is not None and len(mag_curves_to_plot) > 0:
        for i, mag_curve in enumerate(mag_curves_to_plot):
            if np.all(np.isfinite(mag_curve)):
                label = 'MCMC samples' if i == 0 else None
                axes[0].plot(phase_smooth, mag_curve, '-', linewidth=0.5, 
                            color='#CD5C5C', alpha=0.15, zorder=1, label=label)
    
    # Mediana de curvas (línea verde SÓLIDA) - siempre en el centro del CI
    if mag_model_curves is not None:
        axes[0].plot(phase_smooth, mag_model_curves, '-', linewidth=2.5, 
                    label='Median of Curves', zorder=5, color='#1B4332')
    
    # Mediana de parámetros (línea azul PUNTEADA) - puede diferir por correlaciones
    if mag_model_params is not None:
        axes[0].plot(phase_smooth, mag_model_params, '--', linewidth=2.0, 
                    label='Median of Params', zorder=4, color='#4169E1')
    
    # 4. Observaciones normales
    if len(phase_normal_mag) > 0:
        if mag_err_normal is not None and not np.all(np.isnan(mag_err_normal)):
            axes[0].errorbar(phase_normal_mag, mag_normal, yerr=mag_err_normal, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB',
                             capsize=2, capthick=1, elinewidth=1.5)
        else:
            axes[0].errorbar(phase_normal_mag, mag_normal, yerr=None, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB')
    
    # 5. Upper limits usados en el fit
    if len(phase_ul_fit_mag) > 0:
        axes[0].scatter(phase_ul_fit_mag, mag_ul_fit, marker='^', color='green', alpha=0.8, 
                       s=80, label='Upper limits (used in fit)', zorder=9, edgecolors='darkgreen', linewidths=1)
        for px, py in zip(phase_ul_fit_mag, mag_ul_fit):
            axes[0].annotate('', xy=(px, py), xytext=(px, py + 0.3),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8, lw=1.5))
    
    # 6. Upper limits NO usados en el fit
    if phase_ul is not None and mag_ul is not None and len(phase_ul) > 0:
        axes[0].scatter(phase_ul, mag_ul, marker='v', color='orange', alpha=0.7, 
                       s=60, label='Upper limits (not in fit)', zorder=8)
        for px, py in zip(phase_ul, mag_ul):
            axes[0].annotate('', xy=(px, py), xytext=(px, py + 0.3),
                            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
    
    # Configurar eje de magnitud
    mag_all = [mag]
    if mag_model_curves is not None:
        mag_all.append(mag_model_curves)
    if mag_model_params is not None:
        mag_all.append(mag_model_params)
    
    mag_combined = np.concatenate([m for m in mag_all if len(m) > 0])
    if len(mag_combined) > 0:
        mag_min = np.nanmin(mag_combined)
        mag_max = np.nanmax(mag_combined)
        mag_range = mag_max - mag_min
        axes[0].set_ylim(mag_min - 0.1 * mag_range, mag_max + 0.1 * mag_range)
    
    axes[0].set_ylabel('Magnitude', fontsize=10)
    axes[0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(False)  # A&A style: no background grid
    axes[0].tick_params(direction='in', which='both', top=True, right=True)
    axes[0].tick_params(axis='x', labelbottom=True)
    axes[0].invert_yaxis()
    
    # ==========================================================================
    # PLOT DE FLUJO
    # ==========================================================================
    
    # Separar observaciones normales de upper limits (para flujo)
    if is_upper_limit is not None and np.any(is_upper_limit):
        mask_normal = ~is_upper_limit
        mask_ul_fit = is_upper_limit
        
        phase_normal_flux = phase[mask_normal]
        flux_normal = flux[mask_normal]
        flux_err_normal = flux_err[mask_normal] if flux_err is not None and len(flux_err) == len(flux) else None
        
        phase_ul_fit_flux = phase[mask_ul_fit]
        flux_ul_fit = flux[mask_ul_fit]
    else:
        phase_normal_flux = phase
        flux_normal = flux
        flux_err_normal = flux_err if (flux_err is not None and len(flux_err) == len(flux)) else None
        phase_ul_fit_flux = np.array([])
        flux_ul_fit = np.array([])
    
    # Curvas individuales (líneas rojas semitransparentes) - plotear primero (zorder bajo)
    if flux_curves_to_plot is not None and len(flux_curves_to_plot) > 0:
        for i, flux_curve in enumerate(flux_curves_to_plot):
            if np.all(np.isfinite(flux_curve)):
                label = 'MCMC samples' if i == 0 else None
                axes[1].plot(phase_smooth, flux_curve, '-', linewidth=0.5, 
                            color='#CD5C5C', alpha=0.15, zorder=1, label=label)
    
    # Mediana de curvas (línea verde SÓLIDA) - siempre en el centro del CI
    if flux_model_curves is not None:
        axes[1].plot(phase_smooth, flux_model_curves, '-', linewidth=2.5, 
                    label='Median of Curves', zorder=5, color='#1B4332')
    
    # Mediana de parámetros (línea azul PUNTEADA) - puede diferir por correlaciones
    if flux_model_params is not None:
        axes[1].plot(phase_smooth, flux_model_params, '--', linewidth=2.0, 
                    label='Median of Params', zorder=4, color='#4169E1')
    
    # 4. Observaciones normales
    if len(phase_normal_flux) > 0:
        if flux_err_normal is not None and not np.all(np.isnan(flux_err_normal)):
            axes[1].errorbar(phase_normal_flux, flux_normal, yerr=flux_err_normal, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB',
                             capsize=2, capthick=1, elinewidth=1.5)
        else:
            axes[1].errorbar(phase_normal_flux, flux_normal, yerr=None, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB')
    
    # 5. Upper limits usados en el fit
    if len(phase_ul_fit_flux) > 0:
        axes[1].scatter(phase_ul_fit_flux, flux_ul_fit, marker='^', color='green', alpha=0.8, 
                       s=80, label='Upper limits (used in fit)', zorder=9, edgecolors='darkgreen', linewidths=1)
        for px, py in zip(phase_ul_fit_flux, flux_ul_fit):
            max_flux = flux.max() if len(flux) > 0 else flux_ul_fit.max()
            arrow_length = max_flux * 0.05
            axes[1].annotate('', xy=(px, py), xytext=(px, py - arrow_length),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8, lw=1.5))
    
    # 6. Upper limits NO usados en el fit
    if phase_ul is not None and flux_ul is not None and len(phase_ul) > 0:
        axes[1].scatter(phase_ul, flux_ul, marker='v', color='orange', alpha=0.7, 
                       s=60, label='Upper limits (not in fit)', zorder=8)
        for px, py in zip(phase_ul, flux_ul):
            max_flux = flux.max() if len(flux) > 0 else flux_ul.max()
            arrow_length = max_flux * 0.05
            axes[1].annotate('', xy=(px, py), xytext=(px, py - arrow_length),
                            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
    
    # Detectar si phase es MJD o fase relativa
    is_mjd = len(phase) > 0 and phase.min() > 50000
    xlabel = 'MJD' if is_mjd else 'Phase (days)'
    
    # Configurar eje de flujo
    flux_all = [flux]
    if flux_model_curves is not None:
        flux_all.append(flux_model_curves)
    if flux_model_params is not None:
        flux_all.append(flux_model_params)
    
    flux_combined = np.concatenate([f for f in flux_all if len(f) > 0])
    if len(flux_combined) > 0:
        flux_min = np.nanmin(flux_combined)
        flux_max = np.nanmax(flux_combined)
        flux_range = flux_max - flux_min
        axes[1].set_ylim(max(0, flux_min - 0.1 * flux_range), flux_max + 0.1 * flux_range)
    
    # Actualizar labels de los subplots (comparten el eje X)
    # Forzar mostrar ticks y labels en ambos subplots
    axes[0].tick_params(axis='x', labelbottom=True, bottom=True)
    axes[0].set_xlabel(xlabel, fontsize=10)
    axes[1].set_xlabel(xlabel, fontsize=10)
    axes[1].set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$)', fontsize=9)
    axes[1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(False)  # A&A style: no background grid
    axes[1].tick_params(direction='in', which='both', top=True, right=True)
    
    # Aplicar límites comunes del eje X si se proporcionan
    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
    
    plt.tight_layout()
    
    # Retornar los sample_indices usados y el índice del sample de la curva central
    # para reutilizarlos en plot_extended_model
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
        plt.close(fig)  # Liberar memoria de la figura
        return None, sample_indices, best_sample_global_idx
    else:
        return fig, sample_indices, best_sample_global_idx

def plot_extended_model(phase, flux, param_medians, is_upper_limit=None,
                       flux_err=None, sn_name=None, filter_name=None, save_path=None,
                       early_time_offset=-500, late_time_offset=500, samples=None,
                       precalculated_sample_indices=None, central_curve_sample_idx=None):
    """
    Generar gráfico del modelo extendido para validación física
    
    Parameters:
    -----------
    phase : array
        Fases (MJD o fase relativa) de las detecciones normales
    flux : array
        Flujos observados (solo para identificar primera y última detección)
    param_medians : array
        Parámetros medianos del modelo MCMC (fallback si no hay samples)
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos son upper limits
    sn_name, filter_name : str
        Identificadores
    save_path : str, optional
        Ruta para guardar figura
    early_time_offset : float
        Días antes de la primera detección para evaluar (default: -500)
    late_time_offset : float
        Días después de la última detección para evaluar (default: +500)
    samples : array, optional
        Samples del MCMC para calcular mediana de curvas (consistente con fit principal)
    central_curve_sample_idx : int, optional
        Índice del sample que corresponde a la curva central (del plot_fit_with_uncertainty)
        Si se proporciona, se usa directamente en vez de recalcular
    """
    from model import alerce_model
    
    # Identificar primera y última detección normal
    if is_upper_limit is not None and np.any(is_upper_limit):
        mask_normal = ~is_upper_limit
        phase_normal = phase[mask_normal]
    else:
        phase_normal = phase
    
    if len(phase_normal) == 0:
        print(f"    [WARNING] No hay detecciones normales para generar modelo extendido")
        return None
    
    first_phase = phase_normal.min()
    last_phase = phase_normal.max()
    
    # Rango extendido: -500 días antes de primera detección, +500 días después de última
    phase_extended = np.linspace(first_phase + early_time_offset, 
                                 last_phase + late_time_offset, 500)
    
    try:
        # Calcular mediana de parámetros (siempre)
        flux_model_params = alerce_model(phase_extended, *param_medians)
        flux_model_params = np.clip(flux_model_params, 1e-10, None)
        
        # Calcular mediana de curvas (curva verde)
        # IMPORTANTE: Usar EXACTAMENTE el mismo sample que en plot_fit_with_uncertainty
        flux_model_curves = None
        if samples is not None and len(samples) > 0:
            # Si tenemos el índice del sample de la curva central, usarlo directamente
            if central_curve_sample_idx is not None:
                try:
                    flux_model_curves = alerce_model(phase_extended, *samples[central_curve_sample_idx])
                    flux_model_curves = np.clip(flux_model_curves, 1e-10, None)
                    if not (np.all(np.isfinite(flux_model_curves)) and np.all(flux_model_curves > 0)):
                        flux_model_curves = None
                        print(f"    [DEBUG] Modelo extendido: curva central inválida, usando fallback")
                    else:
                        print(f"    [DEBUG] Modelo extendido: usando curva central del plot normal (sample idx={central_curve_sample_idx})")
                except Exception as e:
                    print(f"    [DEBUG] Modelo extendido: error con curva central, usando fallback: {e}")
                    flux_model_curves = None
            
            # Fallback: recalcular si no tenemos el índice o falló
            if flux_model_curves is None:
                # Usar índices precalculados si están disponibles
                if precalculated_sample_indices is not None:
                    sample_indices = precalculated_sample_indices[::10]
                    print(f"    [DEBUG] Modelo extendido (fallback): usando {len(sample_indices)} sample_indices")
                else:
                    n_samples_for_median = min(50, len(samples))
                    step = max(1, len(samples) // n_samples_for_median)
                    sample_indices = np.arange(0, len(samples), step)[:n_samples_for_median]
                
                all_flux_curves = []
                for idx in sample_indices:
                    try:
                        flux_curve = alerce_model(phase_extended, *samples[idx])
                        flux_curve = np.clip(flux_curve, 1e-10, None)
                        if np.all(np.isfinite(flux_curve)) and np.all(flux_curve > 0) and np.all(flux_curve < 1e10):
                            all_flux_curves.append(flux_curve)
                    except:
                        continue
                
                if len(all_flux_curves) >= 10:
                    all_flux_curves = np.array(all_flux_curves)
                    flux_p50 = np.percentile(all_flux_curves, 50, axis=0)
                    distances = np.sum((all_flux_curves - flux_p50)**2, axis=1)
                    best_curve_idx = np.argmin(distances)
                    flux_model_curves = all_flux_curves[best_curve_idx]
                    print(f"    [DEBUG] Modelo extendido (fallback): {len(all_flux_curves)} curvas válidas")
                    del all_flux_curves
        
        # Crear figura separada
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_CONFIG["figsize"][0], PLOT_CONFIG["figsize"][1] * 0.55))
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.97)
        fig.suptitle(f'{sn_name} - {filter_name} - Extended', 
                     fontsize=9, fontweight='bold', y=0.96)
        
        # Mediana de curvas (línea verde SÓLIDA)
        if flux_model_curves is not None:
            ax.plot(phase_extended, flux_model_curves, '-', linewidth=1.5, 
                    color='#1B4332', alpha=0.9, label='Med. Curves', zorder=5)
        
        # Mediana de parámetros (línea azul PUNTEADA)
        ax.plot(phase_extended, flux_model_params, '--', linewidth=1.2, 
                color='#4169E1', alpha=0.9, label='Med. Params', zorder=4)
        
        # Plotear también los datos observados para que el ylim los incluya
        # Usar la misma simbología que en los plots de fit (errorbar)
        if len(flux) > 0:
            # Separar datos normales de upper limits si aplica
            if is_upper_limit is not None and np.any(is_upper_limit):
                mask_normal = ~is_upper_limit
                phase_normal = phase[mask_normal]
                flux_normal = flux[mask_normal]
                flux_err_normal = flux_err[mask_normal] if flux_err is not None else None
            else:
                phase_normal = phase
                flux_normal = flux
                flux_err_normal = flux_err
            
            # Plot observaciones normales con errorbar (misma simbología que plot normal)
            if len(phase_normal) > 0:
                if flux_err_normal is not None and not np.all(np.isnan(flux_err_normal)):
                    ax.errorbar(phase_normal, flux_normal, yerr=flux_err_normal, fmt='o', alpha=0.7,
                               label='Obs.', markersize=3, zorder=10, color='#2E86AB',
                               capsize=1, capthick=0.5, elinewidth=0.8)
                else:
                    ax.errorbar(phase_normal, flux_normal, yerr=None, fmt='o', alpha=0.7,
                               label='Obs.', markersize=3, zorder=10, color='#2E86AB')
        
        # Marcar rango observado con sombreado
        ax.axvspan(first_phase, last_phase, alpha=0.15, color='green', 
                   label='Obs. range', zorder=1)
        
        # Líneas verticales en primera y última detección
        ax.axvline(first_phase, color='green', linestyle='--', linewidth=0.8, alpha=0.5, zorder=3)
        ax.axvline(last_phase, color='green', linestyle='--', linewidth=0.8, alpha=0.5, zorder=3)
        
        # Marcar puntos de validación
        flux_first = alerce_model(np.array([first_phase]), *param_medians)[0]
        flux_early = alerce_model(np.array([first_phase + early_time_offset]), *param_medians)[0]
        flux_last = alerce_model(np.array([last_phase]), *param_medians)[0]
        flux_late = alerce_model(np.array([last_phase + late_time_offset]), *param_medians)[0]
        
        # Puntos de validación (sin labels para reducir leyenda)
        ax.plot([first_phase + early_time_offset], [flux_early], 'ro', markersize=6, zorder=10)
        ax.plot([first_phase], [flux_first], 'go', markersize=6, zorder=10)
        ax.plot([last_phase], [flux_last], 'go', markersize=6, zorder=10)
        ax.plot([last_phase + late_time_offset], [flux_late], 'ro', markersize=6, zorder=10)
        
        # Líneas de conexión para mostrar la validación
        ax.plot([first_phase + early_time_offset, first_phase], [flux_early, flux_first], 
                'r--', linewidth=0.8, alpha=0.4, zorder=4)
        ax.plot([last_phase, last_phase + late_time_offset], [flux_last, flux_late], 
                'r--', linewidth=0.8, alpha=0.4, zorder=4)
        
        # Detectar si phase es MJD (valores grandes > 50000) o fase relativa
        is_mjd = len(phase) > 0 and phase.min() > 50000
        xlabel = 'MJD' if is_mjd else 'Phase (days)'
        
        # Configurar el plot
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$)', fontsize=8)
        ax.grid(False)  # A&A style: no background grid
        ax.legend(loc='upper right', fontsize=6, frameon=True, framealpha=0.9)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        
        # Dejar que matplotlib maneje el ylim automáticamente
        # No ajustar manualmente para evitar problemas con valores pequeños (1e-8)
        
        # Ajustar xlim para mostrar todo el rango extendido
        ax.set_xlim(first_phase + early_time_offset, last_phase + late_time_offset)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
            plt.close(fig)
            return None
        else:
            return fig
            
    except Exception as e:
        print(f"    [ERROR] Error al generar modelo extendido: {str(e)}")
        return None

def plot_corner(samples, param_names=None, save_path=None, param_medians=None, param_percentiles=None):
    """
    Generar corner plot de los parámetros MCMC
    
    NOTA: Usa un subconjunto de samples para DIBUJAR (por memoria), pero muestra
    los valores de param_medians/param_percentiles (de TODOS los samples) en los títulos.
    
    Parameters:
    -----------
    samples : array (n_samples, n_params)
        Samples del MCMC (después de burn-in)
    param_names : list, optional
        Nombres de parámetros
    save_path : str, optional
        Ruta para guardar figura
    param_medians : array, optional
        Mediana de TODOS los samples (de mcmc_results['params'])
    param_percentiles : array (3, n_params), optional
        Percentiles [16, 50, 84] de TODOS los samples (de mcmc_results['params_percentiles'])
    """
    try:
        import corner
    except ImportError:
        import warnings
        warnings.warn("corner no instalado. Instalar con: pip install corner o conda install -c conda-forge corner")
        return None
    
    if param_names is None:
        # Nombres sin itálica: texto plano excepto gamma (símbolo griego)
        param_names = ['A', 'f', r'$t_0$', r'$t_{rise}$', r'$t_{fall}$', r'$\gamma$']
    
    # Limitar número de samples para evitar problemas de memoria
    max_samples_for_corner = 5000  # Máximo 5,000 samples
    if len(samples) > max_samples_for_corner:
        # Muestreo sistemático para representatividad (cada N samples)
        step = len(samples) // max_samples_for_corner
        indices = np.arange(0, len(samples), step)[:max_samples_for_corner]
        samples = samples[indices]
        print(f"    [DEBUG] Corner plot: usando {len(indices)} samples (muestreo sistemático cada {step})")
    
    # Verificar que hay suficientes samples
    if len(samples) < 10:
        import warnings
        warnings.warn(f"Solo hay {len(samples)} samples, el corner plot puede no ser confiable")
    
    # Suprimir warnings específicos de corner sobre contornos
    import warnings
    import logging
    
    # Suprimir warnings de corner y logging
    old_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            # Suprimir todos los warnings de RuntimeWarning y UserWarning durante la creación del corner plot
            warnings.filterwarnings("ignore", message=".*Too few points.*")
            warnings.filterwarnings("ignore", message=".*too few.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # También suprimir warnings de logging
            import sys
            old_stderr = sys.stderr
            from io import StringIO
            sys.stderr = StringIO()
            
            try:
                # Calcular quantiles manualmente para tener control sobre el formato
                quantiles_to_show = [0.16, 0.5, 0.84]  # Percentiles 16, 50 (mediana), 84
                
                # Función para formatear números pequeños
                def format_small_number(val, err_low=None, err_high=None):
                    """Formatear número, usando notación científica si es muy pequeño"""
                    abs_val = abs(val)
                    
                    # Determinar formato basado en magnitud
                    if abs_val < 0.01 or abs_val > 1000 or abs_val == 0:
                        # Notación científica
                        val_str = f"{val:.2e}"
                        if err_low is not None and err_high is not None:
                            err_low_str = f"{abs(err_low):.2e}" if abs(err_low) > 1e-10 else "0.00e+00"
                            err_high_str = f"{err_high:.2e}" if abs(err_high) > 1e-10 else "0.00e+00"
                            return val_str, err_low_str, err_high_str
                        return val_str
                    else:
                        # Determinar decimales basado en incertidumbre
                        if err_low is not None and err_high is not None:
                            max_err = max(abs(err_low), err_high)
                            if max_err < 1e-10:
                                decimals = 6
                            elif max_err < 0.0001:
                                decimals = 6
                            elif max_err < 0.001:
                                decimals = 5
                            elif max_err < 0.01:
                                decimals = 4
                            elif max_err < 0.1:
                                decimals = 3
                            else:
                                decimals = 2
                            
                            val_str = f"{val:.{decimals}f}"
                            err_low_str = f"{abs(err_low):.{decimals}f}" if abs(err_low) > 1e-10 else f"0.{'0'*decimals}"
                            err_high_str = f"{err_high:.{decimals}f}" if abs(err_high) > 1e-10 else f"0.{'0'*decimals}"
                            return val_str, err_low_str, err_high_str
                        else:
                            decimals = 4 if abs_val < 1 else 2
                            return f"{val:.{decimals}f}"
                
                # Calcular rangos más amplios (0.5% - 99.5%) para no cortar distribuciones
                ranges = []
                for i in range(samples.shape[1]):
                    p_low = np.percentile(samples[:, i], 0.5)
                    p_high = np.percentile(samples[:, i], 99.5)
                    margin = (p_high - p_low) * 0.1  # 10% de margen extra
                    ranges.append((p_low - margin, p_high + margin))
                
                # Generar corner plot con tamaño reducido para evitar problemas de memoria
                fig = corner.corner(samples, labels=param_names, show_titles=True,
                                    title_kwargs={"fontsize": 8},
                                    range=ranges,
                                    fig=plt.figure(figsize=(8, 8)))
                
                # Modificar los títulos después para mostrar números pequeños correctamente
                axes = fig.get_axes()
                n_params = len(param_names)
                
                # Los títulos están en los subplots de la diagonal (cada n_params+1)
                for i, param_name in enumerate(param_names):
                    # El subplot diagonal está en la posición i*(n_params+1)
                    ax_idx = i * (n_params + 1)
                    if ax_idx < len(axes):
                        ax = axes[ax_idx]
                        
                        # Usar valores de TODOS los samples si están disponibles
                        if param_medians is not None and param_percentiles is not None:
                            # Usar valores precalculados de mcmc_results (TODOS los samples)
                            median = param_medians[i]
                            p16 = param_percentiles[0, i]  # percentil 16
                            p84 = param_percentiles[2, i]  # percentil 84
                        else:
                            # Fallback: calcular desde el subconjunto de samples
                            param_samples = samples[:, i]
                            median = np.median(param_samples)
                            p16 = np.percentile(param_samples, 16)
                            p84 = np.percentile(param_samples, 84)
                        
                        err_low = median - p16
                        err_high = p84 - median
                        
                        # DEBUG: Mostrar valores del corner plot
                        if i == 0:  # Solo imprimir una vez al inicio
                            source = "mcmc_results" if param_medians is not None else "subconjunto"
                            print(f"    [DEBUG] Valores del CORNER PLOT (de {source}):")
                        print(f"      {param_name} = {median:.6e} (p16={p16:.6e}, p84={p84:.6e})")
                        
                        # Formatear con nuestra función
                        val_str, err_low_str, err_high_str = format_small_number(median, err_low, err_high)
                        
                        # Construir nuevo título de manera segura (sin usar .format() con llaves)
                        new_title = param_name + " = " + val_str + "$"
                        new_title += "^{+" + err_high_str + "}_{-" + err_low_str + "}$"
                        
                        ax.set_title(new_title)
            finally:
                sys.stderr = old_stderr
                
    except Exception as e:
        import warnings
        warnings.warn(f"Error al generar corner plot: {e}")
        return None
    finally:
        logging.root.setLevel(old_level)
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
        plt.close(fig)  # Liberar memoria de la figura
        return None
    else:
        return fig

def plot_prior_likelihood_posterior(samples, times, flux, flux_err, is_upper_limit=None,
                                     dynamic_bounds=None, sn_name=None, filter_name=None, 
                                     save_path=None, n_samples_to_plot=2000):
    """
    Generar gráficos de contornos 2D del prior, likelihood y posterior en el espacio de parámetros
    Similar a la Figura 8.1 de Ivezić et al. (2014), mostrando cómo el prior "corta" el espacio
    
    Parameters:
    -----------
    samples : array (n_samples, n_params)
        Samples del MCMC
    times, flux, flux_err : arrays
        Datos usados en el ajuste
    is_upper_limit : array of bool, optional
        Máscara que indica qué puntos son upper limits
    dynamic_bounds : dict, optional
        Bounds dinámicos usados en el ajuste
    sn_name, filter_name : str, optional
        Identificadores
    save_path : str, optional
        Ruta para guardar figura
    n_samples_to_plot : int
        Número de samples a usar para el cálculo (para no sobrecargar)
    """
    from mcmc_fitter import log_prior, log_likelihood, log_posterior
    from scipy.stats import gaussian_kde
    
    param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
    
    # Subsampling si hay muchos samples
    if len(samples) > n_samples_to_plot:
        indices = np.linspace(0, len(samples)-1, n_samples_to_plot, dtype=int)
        samples_to_use = samples[indices]
    else:
        samples_to_use = samples
    
    # Calcular prior, likelihood y posterior para cada sample
    priors = []
    likelihoods = []
    posteriors = []
    
    for sample in samples_to_use:
        try:
            lp = log_prior(sample, dynamic_bounds, times=times, flux=flux, is_upper_limit=is_upper_limit)
            ll = log_likelihood(sample, times, flux, flux_err, dynamic_bounds, is_upper_limit)
            post = log_posterior(sample, times, flux, flux_err, dynamic_bounds, is_upper_limit)
            
            priors.append(lp if np.isfinite(lp) else -np.inf)
            likelihoods.append(ll if np.isfinite(ll) else -np.inf)
            posteriors.append(post if np.isfinite(post) else -np.inf)
        except:
            priors.append(-np.inf)
            likelihoods.append(-np.inf)
            posteriors.append(-np.inf)
    
    priors = np.array(priors)
    likelihoods = np.array(likelihoods)
    posteriors = np.array(posteriors)
    
    # Filtrar samples válidos
    valid_mask = np.isfinite(posteriors)
    samples_valid = samples_to_use[valid_mask]
    posteriors_valid = posteriors[valid_mask]
    priors_valid = priors[valid_mask]
    likelihoods_valid = likelihoods[valid_mask]
    
    if len(samples_valid) < 10:
        # Si hay muy pocos samples válidos, usar histogramas simples
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        axes[0].hist(priors[valid_mask], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Prior (too few valid samples)')
        axes[1].hist(likelihoods[valid_mask], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title('Likelihood (too few valid samples)')
        axes[2].hist(posteriors_valid, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[2].set_title('Posterior (too few valid samples)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
            plt.close(fig)
            return None
        else:
            return fig
    
    # Seleccionar pares de parámetros importantes para visualizar
    # (A, t0) y (t_rise, t_fall) son los más informativos
    param_pairs = [
        (0, 2, 'A', 't0'),  # A vs t0
        (3, 4, 't_rise', 't_fall'),  # t_rise vs t_fall
    ]
    
    # Crear figura con subplots: una fila por cada par de parámetros, 3 columnas (prior, likelihood, posterior)
    fig, axes = plt.subplots(len(param_pairs), 3, figsize=(15, 5 * len(param_pairs)))
    if len(param_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (idx1, idx2, name1, name2) in enumerate(param_pairs):
        param1_vals = samples_valid[:, idx1]
        param2_vals = samples_valid[:, idx2]
        
        # Rango para los contornos
        x_range = np.linspace(param1_vals.min(), param1_vals.max(), 50)
        y_range = np.linspace(param2_vals.min(), param2_vals.max(), 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Prior contours - usar hexbin para eficiencia
        ax_prior = axes[row, 0]
        prior_valid = priors_valid > -1e6
        if np.sum(prior_valid) > 10:
            # Hexbin plot con colores según prior
            hb_prior = ax_prior.hexbin(param1_vals[prior_valid], param2_vals[prior_valid], 
                                       C=priors_valid[prior_valid], cmap='Blues', gridsize=20, mincnt=1)
            ax_prior.scatter(param1_vals[~prior_valid], param2_vals[~prior_valid], 
                           c='red', marker='x', s=20, alpha=0.5, label='Rejected (exceeds UL)')
        else:
            ax_prior.scatter(param1_vals, param2_vals, c=priors_valid, cmap='Blues', alpha=0.5, s=10)
        ax_prior.set_xlabel(name1)
        ax_prior.set_ylabel(name2)
        ax_prior.set_title(f'Prior: {name1} vs {name2}')
        ax_prior.grid(True, alpha=0.3)
        if np.sum(~prior_valid) > 0:
            ax_prior.legend(fontsize=8)
        
        # Likelihood contours
        ax_likelihood = axes[row, 1]
        likelihood_valid = likelihoods_valid > -1e6
        if np.sum(likelihood_valid) > 10:
            hb_likelihood = ax_likelihood.hexbin(param1_vals[likelihood_valid], param2_vals[likelihood_valid],
                                                C=likelihoods_valid[likelihood_valid], cmap='Greens', gridsize=20, mincnt=1)
        else:
            ax_likelihood.scatter(param1_vals, param2_vals, c=likelihoods_valid, cmap='Greens', alpha=0.5, s=10)
        ax_likelihood.set_xlabel(name1)
        ax_likelihood.set_ylabel(name2)
        ax_likelihood.set_title(f'Likelihood: {name1} vs {name2}')
        ax_likelihood.grid(True, alpha=0.3)
        
        # Posterior contours (el más importante) - usar KDE para contornos suaves
        ax_posterior = axes[row, 2]
        posterior_valid = posteriors_valid > -1e6
        if np.sum(posterior_valid) > 50:
            try:
                # Usar KDE para contornos suaves
                kde_posterior = gaussian_kde(np.vstack([param1_vals[posterior_valid], param2_vals[posterior_valid]]))
                # Evaluar en grilla más pequeña para eficiencia
                x_grid = np.linspace(param1_vals.min(), param1_vals.max(), 30)
                y_grid = np.linspace(param2_vals.min(), param2_vals.max(), 30)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                Z_posterior = kde_posterior(positions).reshape(X_grid.shape)
                
                # Contornos rellenos y líneas
                ax_posterior.contourf(X_grid, Y_grid, Z_posterior, levels=8, cmap='Purples', alpha=0.6)
                ax_posterior.contour(X_grid, Y_grid, Z_posterior, levels=5, colors='darkviolet', linewidths=1.5)
            except:
                # Fallback a hexbin
                hb_posterior = ax_posterior.hexbin(param1_vals[posterior_valid], param2_vals[posterior_valid],
                                                   C=posteriors_valid[posterior_valid], cmap='Purples', gridsize=20, mincnt=1)
        else:
            ax_posterior.scatter(param1_vals[posterior_valid], param2_vals[posterior_valid],
                               c=posteriors_valid[posterior_valid], cmap='Purples', alpha=0.6, s=15)
        ax_posterior.set_xlabel(name1)
        ax_posterior.set_ylabel(name2)
        ax_posterior.set_title(f'Posterior: {name1} vs {name2}')
        ax_posterior.grid(True, alpha=0.3)
        
        # Si hay upper limits, mostrar cómo el prior "corta" el espacio
        if is_upper_limit is not None and np.any(is_upper_limit):
            # Marcar regiones donde el prior es muy negativo (excede upper limits)
            # Esto se verá como una "cortada" en el espacio de parámetros
            ax_posterior.text(0.02, 0.98, 'Upper limits activos\n(prior corta espacio)', 
                            transform=ax_posterior.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'Prior, Likelihood and Posterior in Parameter Space{(" - " + sn_name + " " + filter_name) if sn_name else ""}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
        plt.close(fig)
        return None
    else:
        return fig

