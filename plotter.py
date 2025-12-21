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
                              sn_name=None, filter_name=None, save_path=None):
    """
    Generar gráfico de ajuste mostrando múltiples realizaciones del MCMC
    
    Parameters:
    -----------
    phase, mag, mag_err : arrays
        Datos observados (solo puntos de observación)
    mag_model, flux_model : arrays
        Modelo ajustado (mediana) evaluado en los puntos observados
    samples : array (n_samples, n_params)
        TODOS los samples del MCMC (para mediana se usan todos)
    n_samples_to_show : int
        Número de realizaciones a mostrar para visualización (0 = solo mediana)
        NOTA: Las métricas (RMS, MAD, chi2) se calculan con el modelo mediano (todos los samples)
    sn_name, filter_name : str
        Identificadores
    save_path : str, optional
        Ruta para guardar figura
    """
    from model import alerce_model, flux_to_mag
    
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["figsize"], sharex=True)
    fig.suptitle(f'{sn_name} - Filter {filter_name}', fontsize=13, fontweight='bold', y=0.995)
    
    # Crear array de fase denso para curvas suaves (desde min-10 hasta max+10, cada 1 día)
    phase_min = phase.min()
    phase_max = phase.max()
    phase_smooth = np.arange(phase_min - 10, phase_max + 10 + 0.5, 1.0)
    
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
            n_samples = min(n_samples_to_show, len(valid_samples))
            
            if n_samples < len(valid_samples):
                # Estrategia: seleccionar samples que cubran el rango de incertidumbre
                # 1. Incluir la mediana (1 sample)
                # 2. Incluir percentiles 16 y 84 (2 samples) 
                # 3. Distribuir el resto uniformemente en el espacio de parámetros
                
                selected_indices = []
                
                # 1. Mediana (siempre incluida)
                distances_to_median = np.sum((valid_samples - param_medians)**2, axis=1)
                median_idx = np.argmin(distances_to_median)
                selected_indices.append(median_idx)
                
                # 2. Percentiles 16 y 84 para cada parámetro (muestran límites del intervalo)
                # Buscar samples cercanos a los percentiles
                for percentile_val, target_percentile in [(16, param_percentiles_16), (84, param_percentiles_84)]:
                    distances_to_percentile = np.sum((valid_samples - target_percentile)**2, axis=1)
                    percentile_idx = np.argmin(distances_to_percentile)
                    if percentile_idx not in selected_indices:
                        selected_indices.append(percentile_idx)
                
                # 3. Distribuir el resto uniformemente en el espacio de parámetros
                # Usar cuantiles para seleccionar samples que cubran el rango
                remaining_slots = n_samples - len(selected_indices)
                if remaining_slots > 0:
                    # Crear índices de cuantiles para seleccionar samples distribuidos
                    quantile_indices = np.linspace(0, len(valid_samples) - 1, remaining_slots + 2, dtype=int)[1:-1]
                    
                    # Para cada cuantil, seleccionar el sample más cercano a ese percentil en el espacio de parámetros
                    for q_idx in quantile_indices:
                        # Calcular distancia de cada sample a la mediana (para ordenar)
                        distances = np.sum((valid_samples - param_medians)**2, axis=1)
                        sorted_by_distance = np.argsort(distances)
                        
                        # Seleccionar sample en la posición del cuantil
                        if q_idx < len(sorted_by_distance):
                            candidate_idx = sorted_by_distance[q_idx]
                            if candidate_idx not in selected_indices:
                                selected_indices.append(candidate_idx)
                    
                    # Si aún faltan, completar aleatoriamente del resto
                    remaining_indices = [i for i in range(len(valid_samples)) if i not in selected_indices]
                    if len(selected_indices) < n_samples and len(remaining_indices) > 0:
                        needed = n_samples - len(selected_indices)
                        # Usar semilla basada en hash para reproducibilidad
                        samples_hash = hash(tuple(valid_samples.flatten()[:100]))
                        np.random.seed(abs(samples_hash) % (2**31))
                        additional = np.random.choice(remaining_indices, min(needed, len(remaining_indices)), replace=False)
                        selected_indices.extend(additional)
                
                # Limitar al número deseado
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
    
    # Plot en magnitud - datos observados
    axes[0].errorbar(phase, mag, yerr=mag_err, fmt='o', alpha=0.7, 
                     label='Observations', markersize=5, zorder=10, 
                     capsize=2, capthick=1, elinewidth=1.5, color='#2E86AB')
    
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
    
    # Plot en flujo - datos observados
    axes[1].errorbar(phase, flux, yerr=None, fmt='o', alpha=0.7,
                     label='Observations', markersize=5, zorder=10, color='#2E86AB')
    
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

