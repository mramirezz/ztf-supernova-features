"""
Generación de gráficos para visualización de ajustes
"""
import matplotlib.pyplot as plt
import matplotlib
# Configuración de estilo para papers científicos (sans-serif)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 13
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['grid.linewidth'] = 0.7
matplotlib.rcParams['axes.spines.top'] = True
matplotlib.rcParams['axes.spines.right'] = True
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
    
    # Plot en magnitud
    axes[0].errorbar(phase, mag, yerr=mag_err, fmt='o', alpha=0.6, 
                     label='Observaciones', markersize=4)
    axes[0].plot(phase, mag_model, 'r-', linewidth=2, label='Modelo MCMC')
    axes[0].set_xlabel('Fase (días)')
    axes[0].set_ylabel('Magnitud')
    axes[0].set_title(f'{sn_name} - Filtro {filter_name} (Magnitud)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot en flujo
    axes[1].errorbar(phase, flux, yerr=None, fmt='o', alpha=0.6,
                     label='Observaciones', markersize=4)
    axes[1].plot(phase, flux_model, 'r-', linewidth=2, label='Modelo MCMC')
    axes[1].set_xlabel('Fase (días)')
    axes[1].set_ylabel('Flujo')
    axes[1].set_title(f'{sn_name} - Filtro {filter_name} (Flujo)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
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
                              is_upper_limit=None, flux_err=None, had_upper_limits=None):
    """
    Generar gráfico de ajuste mostrando múltiples realizaciones del MCMC
    
    Parameters:
    -----------
    phase, mag, mag_err : arrays
        Datos observados (puntos de observación + upper limits usados en fit)
    mag_model, flux_model : arrays
        Modelo ajustado (mediana) evaluado en los puntos observados
    samples : array (n_samples, n_params)
        TODOS los samples del MCMC (para mediana se usan todos)
    n_samples_to_show : int
        Número de realizaciones a mostrar para visualización (0 = solo mediana)
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
    fig.suptitle(f'{sn_name} - Filter {filter_name}', fontsize=13, fontweight='bold', y=0.995)
    
    # Crear array de fase denso para curvas suaves
    # Si había upper limits ANTES de combinarlos (aunque se filtraron después), solo extender 5 días antes
    # Si no había upper limits, extender 10 días antes para contexto
    phase_min = phase.min()
    phase_max = phase.max()
    
    # Usar had_upper_limits si está disponible, sino verificar is_upper_limit
    if had_upper_limits is not None:
        has_ul = had_upper_limits
    else:
        # Fallback: verificar is_upper_limit (pero puede estar vacío si se filtraron)
        has_ul = (is_upper_limit is not None and np.any(is_upper_limit))
    
    # Extensión antes del mínimo: 5 días si había upper limits, 10 días si no
    extension_before = 5.0 if has_ul else 10.0
    extension_after = 10.0  # Siempre 10 días después para mostrar la caída
    
    phase_smooth = np.arange(phase_min - extension_before, phase_max + extension_after + 0.5, 1.0)
    
    # Filtrar y seleccionar samples representativos para visualización
    selected_samples = None
    model_fluxes_smooth = []  # Modelos evaluados en fase suave
    
    if n_samples_to_show > 0 and samples is not None:
        # PRIMERO: Filtrar samples que están dentro de los bounds válidos
        from config import MODEL_CONFIG
        bounds = MODEL_CONFIG["bounds"]
        param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
        
        valid_bounds_mask = np.ones(len(samples), dtype=bool)
        for i, param_name in enumerate(param_names):
            bound_min, bound_max = bounds[param_name]
            param_values = samples[:, i]
            valid_bounds_mask &= (param_values > bound_min) & (param_values < bound_max)
        
        samples_in_bounds = samples[valid_bounds_mask]
        
        if len(samples_in_bounds) == 0:
            # Si no hay samples válidos, usar todos pero con advertencia
            samples_in_bounds = samples
        
        # SEGUNDO: Filtrar samples que generan modelos válidos
        valid_model_samples = []
        valid_model_indices = []
        
        for idx, params in enumerate(samples_in_bounds):
            try:
                # Probar evaluar el modelo en un rango pequeño primero
                test_phase = np.linspace(phase_smooth.min(), phase_smooth.max(), 10)
                test_flux = alerce_model(test_phase, *params)
                
                # Verificar que el modelo sea válido
                if (np.all(np.isfinite(test_flux)) and 
                    np.all(test_flux > 0) and 
                    np.all(test_flux < 1e10)):  # Límite superior razonable
                    valid_model_samples.append(params)
                    valid_model_indices.append(idx)
            except:
                continue
        
        if len(valid_model_samples) == 0:
            # Si no hay samples válidos, usar la mediana solamente
            valid_model_samples = []
        else:
            valid_model_samples = np.array(valid_model_samples)
        
        # TERCERO: De los samples válidos, seleccionar representativos del intervalo de confianza
        if len(valid_model_samples) > 0:
            # Calcular estadísticas de la distribución
            param_medians = np.median(valid_model_samples, axis=0)
            param_percentiles_16 = np.percentile(valid_model_samples, 16, axis=0)
            param_percentiles_84 = np.percentile(valid_model_samples, 84, axis=0)
            
            # Filtrar samples que están dentro del intervalo de confianza razonable (3 sigmas)
            param_stds = np.std(valid_model_samples, axis=0)
            z_scores = np.abs((valid_model_samples - param_medians) / (param_stds + 1e-10))
            valid_mask = np.all(z_scores <= 3, axis=1)  # 3 sigmas para incluir más incertidumbre
            valid_samples = valid_model_samples[valid_mask]
            
            if len(valid_samples) == 0:
                valid_samples = valid_model_samples  # Usar todos si el filtro es muy estricto
            
            # Seleccionar samples que representen el intervalo de confianza
            # ESTRATEGIA MEJORADA: Priorizar samples cercanos a mediana y promedio
            # para que las curvas rojas sean consistentes con las líneas verde y azul
            n_samples = min(n_samples_to_show, len(valid_samples))
            
            if n_samples < len(valid_samples):
                selected_indices = []
                
                # Calcular distancias normalizadas a mediana y promedio
                # Normalizar por la desviación estándar de cada parámetro para que todos tengan el mismo peso
                param_stds = np.std(valid_samples, axis=0)
                param_stds = np.where(param_stds < 1e-10, 1.0, param_stds)  # Evitar división por cero
                
                # Distancias normalizadas a la mediana
                distances_to_median = np.sqrt(np.sum(((valid_samples - param_medians) / param_stds)**2, axis=1))
                
                # Calcular promedio también
                param_means = np.mean(valid_samples, axis=0)
                distances_to_mean = np.sqrt(np.sum(((valid_samples - param_means) / param_stds)**2, axis=1))
                
                # Ordenar samples por cercanía a mediana (más cercanos primero)
                sorted_by_median = np.argsort(distances_to_median)
                
                # Estrategia de selección:
                # 1. Los primeros 20% más cercanos a la mediana (representan el centro de la distribución)
                # 2. Algunos cercanos al promedio (para cubrir esa región también)
                # 3. El resto distribuidos en anillos alrededor de la mediana
                
                n_center = max(1, int(n_samples * 0.2))  # 20% del centro
                n_mean = max(1, int(n_samples * 0.1))    # 10% cercanos al promedio
                n_rest = n_samples - n_center - n_mean    # Resto distribuido
                
                # 1. Seleccionar los más cercanos a la mediana
                for idx in sorted_by_median[:n_center]:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                
                # 2. Seleccionar algunos cercanos al promedio
                sorted_by_mean = np.argsort(distances_to_mean)
                mean_count = 0
                for idx in sorted_by_mean:
                    if idx not in selected_indices and mean_count < n_mean:
                        selected_indices.append(idx)
                        mean_count += 1
                
                # 3. Distribuir el resto en anillos alrededor de la mediana
                # Dividir el rango de distancias en cuantiles y seleccionar uno de cada
                if n_rest > 0 and len(selected_indices) < n_samples:
                    remaining_indices = [i for i in range(len(valid_samples)) if i not in selected_indices]
                    if len(remaining_indices) > 0:
                        # Ordenar los restantes por distancia a mediana
                        remaining_distances = distances_to_median[remaining_indices]
                        remaining_sorted = np.argsort(remaining_distances)
                        
                        # Seleccionar distribuidos en el rango de distancias
                        if n_rest <= len(remaining_sorted):
                            # Dividir en n_rest grupos y tomar uno de cada grupo
                            step = max(1, len(remaining_sorted) // n_rest)
                            for i in range(0, len(remaining_sorted), step):
                                if len(selected_indices) < n_samples:
                                    idx = remaining_indices[remaining_sorted[i]]
                                    selected_indices.append(idx)
                        else:
                            # Si necesitamos más, tomar todos los restantes ordenados
                            for i in remaining_sorted:
                                if len(selected_indices) < n_samples:
                                    idx = remaining_indices[i]
                                    selected_indices.append(idx)
                
                # Asegurar que tenemos exactamente n_samples
                selected_indices = selected_indices[:n_samples]
            else:
                selected_indices = np.arange(len(valid_samples))
            
            selected_samples = valid_samples[selected_indices]
            
            # CUARTO: Pre-calcular todos los modelos en fase suave (más eficiente)
            for params in selected_samples:
                try:
                    model_flux_smooth = alerce_model(phase_smooth, *params)
                    model_flux_smooth = np.clip(model_flux_smooth, 1e-10, None)
                    
                    # Verificación más estricta
                    if (np.all(np.isfinite(model_flux_smooth)) and 
                        np.all(model_flux_smooth > 0) and
                        np.all(model_flux_smooth < 1e10) and
                        np.all(np.isfinite(flux_to_mag(model_flux_smooth)))):
                        model_fluxes_smooth.append(model_flux_smooth)
                except:
                    continue
    
    # Evaluar modelo mediano y promedio en fase suave también
    param_medians = np.median(samples, axis=0) if samples is not None else None
    param_means = np.mean(samples, axis=0) if samples is not None else None
    
    if param_medians is not None:
        mag_model_smooth = flux_to_mag(np.clip(alerce_model(phase_smooth, *param_medians), 1e-10, None))
        flux_model_smooth = alerce_model(phase_smooth, *param_medians)
    
    if param_means is not None:
        mag_model_smooth_mean = flux_to_mag(np.clip(alerce_model(phase_smooth, *param_means), 1e-10, None))
        flux_model_smooth_mean = alerce_model(phase_smooth, *param_means)
    
    # Plotear todas las curvas de magnitud en fase suave (primero, para que queden atrás)
    for model_flux_smooth in model_fluxes_smooth:
        mag_model_smooth_sample = flux_to_mag(model_flux_smooth)
        if not (np.any(np.isnan(mag_model_smooth_sample)) or np.any(np.isinf(mag_model_smooth_sample))):
            axes[0].plot(phase_smooth, mag_model_smooth_sample, 'r-', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Separar observaciones normales de upper limits usados en el fit (para magnitud)
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
    
    # Plot en magnitud - observaciones normales
    if len(phase_normal_mag) > 0:
        if mag_err_normal is not None and not np.all(np.isnan(mag_err_normal)):
            axes[0].errorbar(phase_normal_mag, mag_normal, yerr=mag_err_normal, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB',
                             capsize=2, capthick=1, elinewidth=1.5)
        else:
            axes[0].errorbar(phase_normal_mag, mag_normal, yerr=None, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB')
    
    # Plot upper limits usados en el fit (símbolo diferente: triángulo verde)
    if len(phase_ul_fit_mag) > 0:
        axes[0].scatter(phase_ul_fit_mag, mag_ul_fit, marker='^', color='green', alpha=0.8, 
                       s=80, label='Upper limits (used in fit)', zorder=9, edgecolors='darkgreen', linewidths=1)
        for px, py in zip(phase_ul_fit_mag, mag_ul_fit):
            axes[0].annotate('', xy=(px, py), xytext=(px, py + 0.3),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8, lw=1.5))
    
    # Plot upper limits NO usados en el fit (solo para visualización)
    if phase_ul is not None and mag_ul is not None and len(phase_ul) > 0:
        axes[0].scatter(phase_ul, mag_ul, marker='v', color='orange', alpha=0.7, 
                       s=60, label='Upper limits (not in fit)', zorder=8)
        for px, py in zip(phase_ul, mag_ul):
            axes[0].annotate('', xy=(px, py), xytext=(px, py + 0.3),
                            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
    
    # Modelo mediano en fase suave (más visible)
    if param_medians is not None:
        axes[0].plot(phase_smooth, mag_model_smooth, '-', linewidth=2.5, 
                    label='MCMC Median', zorder=5, color='#1B4332')
    
    # Modelo promedio en fase suave
    if param_means is not None:
        axes[0].plot(phase_smooth, mag_model_smooth_mean, '--', linewidth=2, 
                    label='MCMC Mean', zorder=5, color='#40916C', dashes=(5, 3))
    
    # Calcular ylim basado en datos observados, mediana y promedio (no en líneas rojas)
    mag_all = [mag]
    if param_medians is not None:
        mag_all.append(mag_model_smooth)
    if param_means is not None:
        mag_all.append(mag_model_smooth_mean)
    
    mag_combined = np.concatenate([m for m in mag_all if len(m) > 0])
    if len(mag_combined) > 0:
        mag_min = np.nanmin(mag_combined)  # Valor más pequeño = más brillante
        mag_max = np.nanmax(mag_combined)  # Valor más grande = más débil
        mag_range = mag_max - mag_min
        # Agregar margen del 10% arriba y abajo
        # Con invert_yaxis(), queremos que arriba esté el valor más pequeño (más brillante)
        # y abajo el valor más grande (más débil)
        axes[0].set_ylim(mag_min - 0.1 * mag_range, mag_max + 0.1 * mag_range)
    
    axes[0].set_ylabel('Magnitude', fontsize=12)
    axes[0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[0].tick_params(direction='in', which='both', top=True, right=True)
    axes[0].invert_yaxis()  # Invertir: valores más pequeños (más brillantes) arriba
    
    # Usar los mismos modelos ya calculados para flujo (sin recalcular) - primero
    for model_flux_smooth in model_fluxes_smooth:
        axes[1].plot(phase_smooth, model_flux_smooth, 'r-', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Separar observaciones normales de upper limits usados en el fit (para flujo)
    if is_upper_limit is not None and np.any(is_upper_limit):
        mask_normal = ~is_upper_limit
        mask_ul_fit = is_upper_limit
        
        phase_normal_flux = phase[mask_normal]
        flux_normal = flux[mask_normal]
        if flux_err is not None and len(flux_err) == len(flux):
            flux_err_normal = flux_err[mask_normal]
        else:
            flux_err_normal = None
        
        phase_ul_fit_flux = phase[mask_ul_fit]
        flux_ul_fit = flux[mask_ul_fit]
    else:
        phase_normal_flux = phase
        flux_normal = flux
        flux_err_normal = flux_err if (flux_err is not None and len(flux_err) == len(flux)) else None
        phase_ul_fit_flux = np.array([])
        flux_ul_fit = np.array([])
    
    # Plot en flujo - observaciones normales
    if len(phase_normal_flux) > 0:
        if flux_err_normal is not None and not np.all(np.isnan(flux_err_normal)):
            axes[1].errorbar(phase_normal_flux, flux_normal, yerr=flux_err_normal, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB',
                             capsize=2, capthick=1, elinewidth=1.5)
        else:
            axes[1].errorbar(phase_normal_flux, flux_normal, yerr=None, fmt='o', alpha=0.7,
                             label='Observations', markersize=5, zorder=10, color='#2E86AB')
    
    # Plot upper limits usados en el fit (símbolo diferente: triángulo verde)
    if len(phase_ul_fit_flux) > 0:
        axes[1].scatter(phase_ul_fit_flux, flux_ul_fit, marker='^', color='green', alpha=0.8, 
                       s=80, label='Upper limits (used in fit)', zorder=9, edgecolors='darkgreen', linewidths=1)
        for px, py in zip(phase_ul_fit_flux, flux_ul_fit):
            max_flux = flux.max() if len(flux) > 0 else flux_ul_fit.max()
            arrow_length = max_flux * 0.05
            axes[1].annotate('', xy=(px, py), xytext=(px, py - arrow_length),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8, lw=1.5))
    
    # Plot upper limits NO usados en el fit (solo para visualización)
    if phase_ul is not None and flux_ul is not None and len(phase_ul) > 0:
        axes[1].scatter(phase_ul, flux_ul, marker='v', color='orange', alpha=0.7, 
                       s=60, label='Upper limits (not in fit)', zorder=8)
        for px, py in zip(phase_ul, flux_ul):
            max_flux = flux.max() if len(flux) > 0 else flux_ul.max()
            arrow_length = max_flux * 0.05
            axes[1].annotate('', xy=(px, py), xytext=(px, py - arrow_length),
                            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
    
    # Modelo mediano en fase suave (más visible)
    if param_medians is not None:
        axes[1].plot(phase_smooth, flux_model_smooth, '-', linewidth=2.5, 
                    label='MCMC Median', zorder=5, color='#1B4332')
    
    # Modelo promedio en fase suave
    if param_means is not None:
        axes[1].plot(phase_smooth, flux_model_smooth_mean, '--', linewidth=2, 
                    label='MCMC Mean', zorder=5, color='#40916C', dashes=(5, 3))
    
    # Calcular ylim basado en datos observados, mediana y promedio (no en líneas rojas)
    flux_all = [flux]
    if param_medians is not None:
        flux_all.append(flux_model_smooth)
    if param_means is not None:
        flux_all.append(flux_model_smooth_mean)
    
    flux_combined = np.concatenate([f for f in flux_all if len(f) > 0])
    if len(flux_combined) > 0:
        flux_min = np.nanmin(flux_combined)
        flux_max = np.nanmax(flux_combined)
        flux_range = flux_max - flux_min
        # Agregar margen del 10% arriba y abajo
        axes[1].set_ylim(max(0, flux_min - 0.1 * flux_range), flux_max + 0.1 * flux_range)
    
    axes[1].set_xlabel('Phase (days)', fontsize=12)
    axes[1].set_ylabel('Flux', fontsize=12)
    axes[1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[1].tick_params(direction='in', which='both', top=True, right=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], format=PLOT_CONFIG["format"])
        plt.close(fig)  # Liberar memoria de la figura
        return None
    else:
        return fig

def plot_corner(samples, param_names=None, save_path=None):
    """
    Generar corner plot de los parámetros MCMC
    
    NOTA: Este corner plot muestra TODOS los samples del MCMC, no solo los seleccionados
    para visualización. Esto permite ver la distribución completa de parámetros.
    
    Parameters:
    -----------
    samples : array (n_samples, n_params)
        TODOS los samples del MCMC (después de burn-in)
    param_names : list, optional
        Nombres de parámetros
    save_path : str, optional
        Ruta para guardar figura
    """
    try:
        import corner
    except ImportError:
        import warnings
        warnings.warn("corner no instalado. Instalar con: pip install corner o conda install -c conda-forge corner")
        return None
    
    if param_names is None:
        param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
    
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
                fig = corner.corner(samples, labels=param_names, show_titles=True,
                                    title_kwargs={"fontsize": 10})
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

