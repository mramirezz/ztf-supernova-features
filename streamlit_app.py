"""
App Streamlit para explorar ajustes MCMC de supernovas
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from reader import parse_photometry_file, prepare_lightcurve, mjd_to_phase
from mcmc_fitter import fit_mcmc
from plotter import plot_corner
from feature_extractor import extract_features
from config import BASE_DATA_PATH, PLOTS_DIR, FEATURES_DIR, DATA_FILTER_CONFIG

st.set_page_config(page_title="Explorador MCMC Supernovas", layout="wide")

st.title("Explorador de Ajustes MCMC - Supernovas ZTF")

def _show_parameter_distributions(sn_type):
    """
    Mostrar distribuciones de los 6 parámetros principales (A, f, t0, t_rise, t_fall, gamma)
    separados por filtro
    """
    output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}.csv"
    
    if not output_file.exists():
        st.warning(f"No se encontró el archivo de features: {output_file}")
        st.info("Ejecuta MCMC para al menos un filtro para ver las distribuciones")
        return
    
    try:
        df = pd.read_csv(output_file)
        
        if len(df) == 0:
            st.warning("El archivo de features está vacío")
            return
        
        # Los 6 parámetros principales
        param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
        param_labels = {
            'A': 'A (Amplitude)',
            'f': 'f (Fraction)',
            't0': 't0 (Peak time, days)',
            't_rise': 't_rise (Rise time, days)',
            't_fall': 't_fall (Fall time, days)',
            'gamma': 'gamma (Gamma, days)'
        }
        
        # Obtener filtros únicos
        available_filters = sorted(df['filter_band'].unique())
        
        if len(available_filters) == 0:
            st.warning("No hay datos de filtros disponibles")
            return
        
        st.subheader("Parameter Distributions by Filter")
        st.caption(f"Showing distributions of the 6 main MCMC model parameters for type: **{sn_type}**")
        
        # Crear gráficos: 2 filas x 3 columnas para los 6 parámetros
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Colores representativos para filtros Sloan: g (verde) y r (rojo)
        filter_colors = {
            'g': '#2E7D32',  # Verde para Sloan g
            'r': '#C62828',  # Rojo para Sloan r
            'i': '#7B1FA2',  # Púrpura para Sloan i (por si acaso)
            'z': '#424242'   # Gris oscuro para Sloan z (por si acaso)
        }
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # Plotear histograma para cada filtro
            for i, filter_name in enumerate(available_filters):
                filter_data = df[df['filter_band'] == filter_name][param].values
                
                if len(filter_data) > 0:
                    # Filtrar valores válidos (no NaN, no infinitos)
                    valid_data = filter_data[np.isfinite(filter_data)]
                    
                    if len(valid_data) > 0:
                        # Usar color específico del filtro o un color por defecto
                        color = filter_colors.get(filter_name.lower(), plt.cm.tab10(i))
                        ax.hist(valid_data, bins=min(20, max(5, len(valid_data)//2)), 
                               alpha=0.7, label=f'Filter {filter_name}', 
                               color=color, edgecolor='black', linewidth=0.8)
            
            ax.set_xlabel(param_labels.get(param, param), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution of {param_labels.get(param, param)}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.tick_params(direction='in', which='both', top=True, right=True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Estadísticas por filtro
        st.subheader("Descriptive Statistics by Filter")
        
        for filter_name in available_filters:
            filter_df = df[df['filter_band'] == filter_name]
            
            if len(filter_df) == 0:
                continue
            
            st.markdown(f"### Filter {filter_name}")
            
            # Crear tabla de estadísticas
            stats_data = []
            for param in param_names:
                param_data = filter_df[param].values
                valid_data = param_data[np.isfinite(param_data)]
                
                if len(valid_data) > 0:
                    stats_data.append({
                        'Parameter': param_labels.get(param, param),
                        'Mean': f"{np.mean(valid_data):.4e}" if param == 'A' else f"{np.mean(valid_data):.4f}",
                        'Median': f"{np.median(valid_data):.4e}" if param == 'A' else f"{np.median(valid_data):.4f}",
                        'Std': f"{np.std(valid_data):.4e}" if param == 'A' else f"{np.std(valid_data):.4f}",
                        'Min': f"{np.min(valid_data):.4e}" if param == 'A' else f"{np.min(valid_data):.4f}",
                        'Max': f"{np.max(valid_data):.4e}" if param == 'A' else f"{np.max(valid_data):.4f}",
                        'N': len(valid_data)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, width='stretch')
        
        # Información general
        st.info(f"**Total records**: {len(df)} | **Available filters**: {', '.join(available_filters)} | **Unique supernovae**: {df['sn_name'].nunique()}")
        
    except Exception as e:
        st.error(f"Error loading distributions: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

def _process_single_filter(filters_data, sn_name, filter_name, selected_type,
                          n_walkers, n_steps, burn_in, n_samples_to_show, save_results):
    """Función auxiliar para procesar un solo filtro"""
    try:
        # Preparar datos SIN filtrar para visualización completa (incluyendo upper limits)
        df_all = filters_data[filter_name].copy()
        
        if len(df_all) < 5:
            st.error(f"No hay suficientes datos para filtro {filter_name}")
            return
        
        # Separar datos normales de upper limits
        df_normal = df_all[~df_all['Upperlimit']].copy()
        df_upperlimit = df_all[df_all['Upperlimit']].copy()
        
        # Datos normales - usar MJD directamente para plotting
        mjd_all = df_normal['MJD'].values if len(df_normal) > 0 else np.array([])
        phase_all = mjd_to_phase(mjd_all) if len(df_normal) > 0 else np.array([])  # Para cálculos internos
        mag_all = df_normal['MAG'].values if len(df_normal) > 0 else np.array([])
        mag_err_all = df_normal['MAGERR'].values if len(df_normal) > 0 else np.array([])
        flux_all = 10**(-mag_all / 2.5) if len(df_normal) > 0 else np.array([])
        flux_err_all = (mag_err_all * flux_all) / 1.086 if len(df_normal) > 0 else np.array([])
        
        # Upper limits - usar MJD directamente para plotting
        mjd_ul = df_upperlimit['MJD'].values if len(df_upperlimit) > 0 else np.array([])
        phase_ul = mjd_to_phase(mjd_ul) if len(df_upperlimit) > 0 else np.array([])  # Para cálculos internos
        mag_ul = df_upperlimit['MAG'].values if len(df_upperlimit) > 0 else np.array([])
        flux_ul = 10**(-mag_ul / 2.5) if len(df_upperlimit) > 0 else np.array([])
        
        # Identificar el pico en todos los datos (solo de los normales) - usar MJD
        if len(flux_all) > 0:
            peak_idx_all = np.argmax(flux_all)
            peak_mjd_all = mjd_all[peak_idx_all]
            peak_phase_all = phase_all[peak_idx_all]  # Para labels que mencionan fase
        else:
            peak_mjd_all = None
            peak_phase_all = None
        
        # Preparar datos con filtro temporal (solo hasta 300 días después del pico)
        lc_data = prepare_lightcurve(
            filters_data[filter_name], 
            filter_name,
            max_days_after_peak=DATA_FILTER_CONFIG["max_days_after_peak"],
            max_days_before_peak=DATA_FILTER_CONFIG["max_days_before_peak"],
            max_days_before_first_obs=DATA_FILTER_CONFIG["max_days_before_first_obs"]
        )
        
        if lc_data is None:
            st.error(f"No hay suficientes datos filtrados para filtro {filter_name}")
            return
        
        phase = lc_data['phase']  # Fase relativa para MCMC
        mjd = lc_data.get('mjd', None)  # MJD original para plotting
        reference_mjd = lc_data.get('reference_mjd', None)  # MJD de referencia
        flux = lc_data['flux']
        flux_err = lc_data['flux_err']
        mag = lc_data['mag']
        mag_err = lc_data['mag_err']
        peak_phase = lc_data.get('peak_phase', None)
        is_upper_limit = lc_data.get('is_upper_limit', None)
        had_upper_limits = lc_data.get('had_upper_limits', False)
        
        # Para el "Full Light Curve View", mostrar los upper limits que se usaron en el MCMC
        # Estos están en mjd, mag, flux con is_upper_limit=True
        # También mostrar otros upper limits del archivo que estén cerca de las observaciones
        mjd_ul = np.array([])
        mag_ul = np.array([])
        flux_ul = np.array([])
        
        # Primero, agregar los upper limits que se usaron en el MCMC
        if is_upper_limit is not None and np.any(is_upper_limit) and mjd is not None and len(mjd) > 0:
            mjd_ul_used = mjd[is_upper_limit]
            mag_ul_used = mag[is_upper_limit]
            flux_ul_used = flux[is_upper_limit]
            
            mjd_ul = np.concatenate([mjd_ul, mjd_ul_used]) if len(mjd_ul_used) > 0 else mjd_ul
            mag_ul = np.concatenate([mag_ul, mag_ul_used]) if len(mag_ul_used) > 0 else mag_ul
            flux_ul = np.concatenate([flux_ul, flux_ul_used]) if len(flux_ul_used) > 0 else flux_ul
        
        # También agregar otros upper limits del archivo que estén cerca de las observaciones
        if len(df_upperlimit) > 0 and len(mjd_all) > 0:
            mjd_min_obs = mjd_all.min()
            mjd_max_obs = mjd_all.max()
            margin_days = 100.0
            mjd_ul_all = df_upperlimit['MJD'].values
            
            # Filtrar upper limits razonables (cerca de observaciones y MJD > 50000)
            mask_reasonable = (mjd_ul_all >= mjd_min_obs - margin_days) & (mjd_ul_all <= mjd_max_obs + margin_days)
            mask_reasonable = mask_reasonable & (mjd_ul_all > 50000)
            
            # Excluir los que ya están en mjd_ul (los que se usaron en el fit)
            if len(mjd_ul) > 0:
                mask_reasonable = mask_reasonable & ~np.isin(mjd_ul_all, mjd_ul)
            
            if np.any(mask_reasonable):
                mjd_ul_others = mjd_ul_all[mask_reasonable]
                mag_ul_others = df_upperlimit['MAG'].values[mask_reasonable]
                flux_ul_others = 10**(-mag_ul_others / 2.5)
                
                mjd_ul = np.concatenate([mjd_ul, mjd_ul_others])
                mag_ul = np.concatenate([mag_ul, mag_ul_others])
                flux_ul = np.concatenate([flux_ul, flux_ul_others])
        
        # ===== GRÁFICO DE CONTEXTO: TODA LA DATA =====
        st.subheader(f"Full Light Curve View - Filter {filter_name}")
        
        # Crear gráfico con toda la data - usar MJD para plotting
        fig_all, (ax1_all, ax2_all) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gráfico de magnitud
        if len(mjd_all) > 0:
            ax1_all.errorbar(mjd_all, mag_all, yerr=mag_err_all, fmt='o', 
                            color='#2E86AB', alpha=0.7, markersize=5, label='Detections',
                            capsize=2, capthick=1, elinewidth=1.5, zorder=10)
        
        if len(mjd_ul) > 0:
            ax1_all.scatter(mjd_ul, mag_ul, marker='v', color='orange', alpha=0.7, 
                           s=60, label='Upper limits', zorder=9)
            for px, py in zip(mjd_ul, mag_ul):
                ax1_all.annotate('', xy=(px, py), xytext=(px, py + 0.3),
                                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
        
        if peak_mjd_all is not None:
            ax1_all.axvline(peak_mjd_all, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                           label=f'Flux peak (MJD {peak_mjd_all:.1f})')
        
        if mjd is not None and len(mjd) > 0:
            region_start = mjd.min()
            region_end = mjd.max()
            ax1_all.axvspan(region_start, region_end, alpha=0.2, color='green', 
                          label=f'MCMC region ({len(mjd)} points)')
        
        ax1_all.set_ylabel('Magnitude', fontsize=12)
        ax1_all.set_title(f'{sn_name} - Filter {filter_name} (Full View)', fontsize=13, fontweight='bold')
        ax1_all.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1_all.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1_all.tick_params(direction='in', which='both', top=True, right=True)
        ax1_all.invert_yaxis()
        
        # Gráfico de flujo
        if len(mjd_all) > 0:
            ax2_all.errorbar(mjd_all, flux_all, yerr=flux_err_all, fmt='o', 
                            color='#2E86AB', alpha=0.7, markersize=5, label='Detections',
                            capsize=2, capthick=1, elinewidth=1.5, zorder=10)
        
        if len(mjd_ul) > 0:
            ax2_all.scatter(mjd_ul, flux_ul, marker='v', color='orange', alpha=0.7, 
                           s=60, label='Upper limits', zorder=9)
            for px, py in zip(mjd_ul, flux_ul):
                max_flux = flux_all.max() if len(flux_all) > 0 else flux_ul.max()
                arrow_length = max_flux * 0.05
                ax2_all.annotate('', xy=(px, py), xytext=(px, py - arrow_length),
                                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))
        
        if peak_mjd_all is not None:
            ax2_all.axvline(peak_mjd_all, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                           label=f'Flux peak (MJD {peak_mjd_all:.1f})')
        
        if mjd is not None and len(mjd) > 0:
            region_start = mjd.min()
            region_end = mjd.max()
            ax2_all.axvspan(region_start, region_end, alpha=0.2, color='green', 
                          label=f'MCMC region ({len(mjd)} points)')
        
        ax2_all.set_xlabel('MJD', fontsize=12)
        ax2_all.set_ylabel('Flux', fontsize=12)
        ax2_all.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2_all.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2_all.tick_params(direction='in', which='both', top=True, right=True)
        
        plt.tight_layout()
        st.pyplot(fig_all)
        plt.close()
        
        # Estadísticas
        st.subheader("Estadísticas de los Datos")
        
        # Contar puntos para MCMC
        n_normal_mcmc = np.sum(~is_upper_limit) if is_upper_limit is not None else len(phase)
        n_ul_mcmc = np.sum(is_upper_limit) if is_upper_limit is not None else 0
        
        # Contar cuántas detecciones se descartaron por el filtro temporal
        n_discarded = len(phase_all) - n_normal_mcmc
        
        # Contar upper limits del archivo para ESTE filtro específico que están antes de la primera observación
        # Estos son los que prepare_lightcurve puede usar
        # Usar el mismo DataFrame que se pasó a prepare_lightcurve para asegurar consistencia
        df_this_filter = filters_data[filter_name].copy()
        df_ul_this_filter = df_this_filter[df_this_filter['Upperlimit']].copy()
        
        first_obs_mjd = mjd_all.min() if len(mjd_all) > 0 else None
        n_ul_in_file = 0
        if first_obs_mjd is not None and len(df_ul_this_filter) > 0:
            # Upper limits antes de la primera observación y dentro de 20 días
            ul_before = df_ul_this_filter[(df_ul_this_filter['MJD'] < first_obs_mjd) & 
                                          (df_ul_this_filter['MJD'] >= first_obs_mjd - 20.0)]
            n_ul_in_file = len(ul_before)  # Total disponibles (máximo 3 se usan)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Puntos detectados (sin filtrar)", len(phase_all))
            st.caption("Todas las detecciones normales del archivo")
        with col2:
            st.metric("Upper limits en archivo", len(df_ul_this_filter))
            if n_ul_in_file > 0:
                st.caption(f"Total: {len(df_ul_this_filter)}, {n_ul_in_file} antes de 1ra obs (máx {min(n_ul_in_file, 3)} usados)")
            elif n_ul_mcmc > 0:
                st.caption(f"Total: {len(df_ul_this_filter)}, pero {n_ul_mcmc} usados (revisar)")
            else:
                st.caption(f"Filtro {filter_name}: {len(df_ul_this_filter)} upper limits")
        with col3:
            st.metric("Puntos para MCMC", len(phase))
            caption_parts = []
            if n_discarded > 0:
                caption_parts.append(f"{n_normal_mcmc} de {len(phase_all)} detecciones")
            else:
                caption_parts.append(f"{n_normal_mcmc} detecciones")
            if n_ul_mcmc > 0:
                caption_parts.append(f"+ {n_ul_mcmc} upper limits")
            if n_discarded > 0:
                caption_parts.append(f"({n_discarded} descartadas por filtro)")
            st.caption(" | ".join(caption_parts))
        with col4:
            if mjd_all is not None and len(mjd_all) > 0:
                st.metric("MJD mín", f"{mjd_all.min():.1f}")
            else:
                st.metric("MJD mín", "N/A")
        with col5:
            if mjd_all is not None and len(mjd_all) > 0:
                st.metric("MJD máx", f"{mjd_all.max():.1f}")
            else:
                st.metric("MJD máx", "N/A")
        with col6:
            if mjd_all is not None and len(mjd_all) > 0:
                st.metric("Duración total", f"{mjd_all.max() - mjd_all.min():.1f} días")
            else:
                st.metric("Duración total", "N/A")
        
        # Construir texto de puntos MCMC
        if n_ul_mcmc > 0:
            puntos_text = f"{n_normal_mcmc} detecciones normales filtradas + {n_ul_mcmc} upper limits agregados antes de la primera observación"
        else:
            puntos_text = f"{n_normal_mcmc} detecciones normales filtradas"
        
        # Calcular peak_mjd si tenemos los datos necesarios
        if peak_phase is not None and reference_mjd is not None:
            peak_mjd = reference_mjd + peak_phase
            peak_mjd_str = f"{peak_mjd:.1f}"
            peak_phase_str = f"{peak_phase:.1f} días (días desde la primera detección normal en MJD {reference_mjd:.1f})"
        else:
            peak_mjd_str = "N/A"
            peak_phase_str = "N/A"
        
        # Construir texto del filtro temporal
        if DATA_FILTER_CONFIG['max_days_before_peak'] is None:
            filter_text = f"Solo datos hasta {DATA_FILTER_CONFIG['max_days_after_peak']:.0f} días después del peak (sin límite antes del peak)"
        else:
            filter_text = f"Solo datos entre {DATA_FILTER_CONFIG['max_days_before_peak']:.0f} días antes y {DATA_FILTER_CONFIG['max_days_after_peak']:.0f} días después del peak"
        
        # Explicar fase relativa y filtro
        st.info(f"**Peak en MJD**: {peak_mjd_str} | "
               f"**Fase relativa del peak**: {peak_phase_str} | "
               f"**Filtro temporal**: {filter_text} | "
               f"**Puntos usados para MCMC**: {len(phase)} ({puntos_text})")
        
        st.divider()
        
        # Ajuste MCMC
        t0_mcmc = time.time()
        mcmc_results = fit_mcmc(phase, flux, flux_err, verbose=False, is_upper_limit=is_upper_limit)
        t_mcmc = time.time() - t0_mcmc
        
        from model import flux_to_mag
        mag_model = flux_to_mag(np.clip(mcmc_results['model_flux'], 1e-10, None))
        
        n_total_samples = len(mcmc_results['samples'])
        st.info(f"**Samples totales**: {n_total_samples:,} | **Curvas mostradas**: {n_samples_to_show}")
        
        # Resultados
        st.subheader(f"MCMC Fit Results - Filter {filter_name}")
        st.caption(f"Estos valores son las **medianas de cada parámetro** calculadas por separado de los {n_total_samples:,} samples del MCMC. "
                   f"Se calculan como: mediana_A, mediana_f, mediana_t0, etc. "
                   f"Luego se evalúa el modelo con estos parámetros medianos para obtener la curva 'MCMC Median'.")
        
        param_info = {
            'A': 'Amplitud del flujo máximo (normalización)',
            'f': 'Fracción que controla la forma del peak (0-1)',
            't0': 'Tiempo del peak máximo de flujo (días)',
            't_rise': 'Tiempo característico de subida (días)',
            't_fall': 'Tiempo característico de caída (días)',
            'gamma': 'Parámetro de transición entre fase temprana y tardía (días)'
        }
        
        param_names = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma']
        params = mcmc_results['params']
        params_err = mcmc_results['params_err']
        
        for row in range(2):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < len(param_names):
                    name = param_names[idx]
                    val = params[idx]
                    err = params_err[idx]
                    with cols[col]:
                        st.metric(name, f"{val:.4e}", delta=None)
                        st.caption(f"*{param_info[name]}*")
                        # Calcular porcentaje de incertidumbre relativa
                        rel_uncertainty = (err / abs(val) * 100) if abs(val) > 1e-10 else np.inf
                        st.caption(f"Uncertainty: ±{err:.4e} ({rel_uncertainty:.1f}%)")
        
        # Solo crear carpeta y rutas si se van a guardar resultados
        plot_save_path = None
        corner_save_path = None
        if save_results:
            # Crear subcarpeta para esta supernova solo si se van a guardar
            # Organizar por tipo de supernova: plots/SN Ia/ZTF20abc/
            sn_plots_dir = PLOTS_DIR / selected_type / sn_name
            sn_plots_dir.mkdir(parents=True, exist_ok=True)
            plot_save_path = str(sn_plots_dir / f"{sn_name}_{filter_name}_fit.png")
            corner_save_path = str(sn_plots_dir / f"{sn_name}_{filter_name}_corner.png")
        
        # Convertir modelo a MJD para plotting (igual que en modo debug)
        if mjd is not None and reference_mjd is not None:
            # Ajustar samples: t0 está en fase relativa, convertirlo a MJD absoluto
            from model import alerce_model
            samples_mjd = mcmc_results['samples'].copy()
            samples_mjd[:, 2] = samples_mjd[:, 2] + reference_mjd  # t0 en MJD absoluto
            
            # Recalcular modelo mediano en MJD
            param_medians_mjd = np.median(samples_mjd, axis=0)
            flux_model_points_mjd = alerce_model(mjd, *param_medians_mjd)
            flux_model_points_mjd = np.clip(flux_model_points_mjd, 1e-10, None)
            mag_model_points_mjd = flux_to_mag(flux_model_points_mjd)
            
            # Usar MJD y samples ajustados para el plot
            phase_for_plot = mjd
            mag_model_for_plot = mag_model_points_mjd
            flux_model_for_plot = flux_model_points_mjd
            samples_for_plot = samples_mjd
        else:
            # Fallback: usar fase original
            phase_for_plot = phase
            mag_model_for_plot = mag_model
            flux_model_for_plot = mcmc_results['model_flux']
            samples_for_plot = mcmc_results['samples']
        
        # Los upper limits que se usaron en el fit ya están incluidos en phase, mag, flux con is_upper_limit=True
        # No necesitamos pasar upper limits adicionales al plot, solo los que ya están en los datos del fit
        # Gráficos
        st.subheader("Gráficos de Ajuste")
        from plotter import plot_fit_with_uncertainty
        t0_plot = time.time()
        fig = plot_fit_with_uncertainty(
            phase_for_plot, mag, mag_err, mag_model_for_plot, flux, flux_model_for_plot,
            samples_for_plot, n_samples_to_show,
            sn_name, filter_name, save_path=plot_save_path,
            phase_ul=None,  # No pasar upper limits adicionales, ya están en los datos del fit
            mag_ul=None,
            flux_ul=None,
            is_upper_limit=is_upper_limit, flux_err=flux_err,
            had_upper_limits=had_upper_limits
        )
        t_plot = time.time() - t0_plot
        st.pyplot(fig)
        plt.close()
        
        if save_results and plot_save_path:
                st.success(f"Gráfico guardado: {plot_save_path}")
        
        # Corner plot
        st.subheader("Corner Plot")
        n_samples_corner = len(mcmc_results['samples'])
        st.caption(f"Este corner plot muestra la distribución de los **parámetros** de **{n_samples_corner:,} samples** del MCMC (después de burn-in). Cada sample es un conjunto de 6 parámetros (A, f, t0, t_rise, t_fall, gamma).")
        t0_corner = time.time()
        corner_fig = plot_corner(mcmc_results['samples'], save_path=corner_save_path)
        t_corner = time.time() - t0_corner
        
        if corner_fig is None:
            st.error("No se pudo generar el corner plot.")
            t_corner = 0
        else:
            st.pyplot(corner_fig)
            plt.close()
            if save_results and corner_save_path:
                st.success(f"Corner plot guardado: {corner_save_path}")
        
        # Métricas
        st.subheader("Métricas de Tiempo")
        col_time1, col_time2, col_time3, col_time4 = st.columns(4)
        with col_time1:
            st.metric("MCMC", f"{t_mcmc:.2f} s")
        with col_time2:
            st.metric("Gráfico fit", f"{t_plot:.2f} s")
        with col_time3:
            st.metric("Corner plot", f"{t_corner:.2f} s")
        with col_time4:
            t_total = t_mcmc + t_plot + t_corner
            st.metric("Total", f"{t_total:.2f} s")
        
        # Estadísticas de ajuste
        st.subheader("Estadísticas de Ajuste")
        from feature_extractor import calculate_fit_statistics
        stats = calculate_fit_statistics(phase, flux, flux_err, mcmc_results['model_flux'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Mostrar RMS con formato apropiado para valores pequeños
            if stats['rms'] < 0.0001:
                rms_str = f"{stats['rms']:.2e}"
            else:
                rms_str = f"{stats['rms']:.6f}"
            st.metric("RMS", rms_str)
            with st.expander("Ver ecuación RMS"):
                st.markdown("""
                **Root Mean Square (RMS)** mide la desviación promedio entre los datos observados y el modelo ajustado.
                
                **Nota importante:** Esta métrica se calcula en **flujo**, ya que el MCMC ajusta en flujo. Esto es consistente con el espacio donde se realiza el ajuste.
                
                **Ecuación:**
                $$
                \\text{RMS} = \\sqrt{\\frac{\\sum_{i=1}^{N} (F_i - \\hat{F}_i)^2}{N - p}}
                $$
                
                Donde:
                - $F_i$ = flujo observado en el punto $i$
                - $\\hat{F}_i$ = flujo predicho por el modelo en el punto $i$
                - $N$ = número de puntos observacionales
                - $p$ = número de parámetros del modelo (6: A, f, t₀, t_rise, t_fall, γ)
                - $N - p$ = grados de libertad (degrees of freedom, dof)
                
                **Interpretación:**
                - RMS más pequeño = mejor ajuste
                - Se divide por $(N-p)$ en lugar de $N$ para corregir por el sesgo (Bessel's correction)
                - Unidades: flujo (mismas unidades que los datos observados)
                """)
        
        with col2:
            # Mostrar MAD con formato apropiado para valores pequeños
            if stats['mad'] < 0.0001:
                mad_str = f"{stats['mad']:.2e}"
            else:
                mad_str = f"{stats['mad']:.6f}"
            st.metric("MAD", mad_str)
            with st.expander("Ver ecuación MAD"):
                st.markdown("""
                **Median Absolute Deviation (MAD)** mide la mediana de las desviaciones absolutas. Es más robusto a outliers que el RMS.
                
                **Nota importante:** Esta métrica se calcula en **flujo**, ya que el MCMC ajusta en flujo. Esto es consistente con el espacio donde se realiza el ajuste.
                
                **Ecuación:**
                $$
                \\text{MAD} = \\text{mediana}(|F_i - \\hat{F}_i|)
                $$
                
                Donde:
                - $F_i$ = flujo observado en el punto $i$
                - $\\hat{F}_i$ = flujo predicho por el modelo en el punto $i$
                - Se calcula el valor absoluto de cada residual y luego se toma la mediana
                
                **Interpretación:**
                - MAD más pequeño = mejor ajuste
                - Es resistente a outliers (no se ve afectado por puntos extremos)
                - Unidades: flujo (mismas unidades que los datos observados)
                - Típicamente MAD < RMS cuando hay outliers
                """)
        
        with col3:
            st.metric("Error Relativo Mediano", f"{stats['median_relative_error_pct']:.2f}%")
            with st.expander("Ver ecuación Error Relativo"):
                st.markdown("""
                **Error Relativo Mediano** mide el porcentaje de desviación promedio entre los datos observados y el modelo ajustado, relativo al flujo observado.
                
                **Nota importante:** Esta métrica se calcula en **flujo**, y no depende de los errores observacionales, solo de la diferencia entre observación y modelo.
                
                **Ecuación:**
                $$
                \\text{Error Relativo Mediano} = \\text{mediana}\\left(\\left|\\frac{F_i - \\hat{F}_i}{F_i}\\right| \\times 100\\right)
                $$
                
                Donde:
                - $F_i$ = flujo observado en el punto $i$
                - $\\hat{F}_i$ = flujo predicho por el modelo en el punto $i$
                - Se calcula el error relativo absoluto para cada punto y luego se toma la mediana
                - El resultado se multiplica por 100 para expresarlo como porcentaje
                
                **Interpretación:**
                - Error relativo más pequeño = mejor ajuste
                - Es independiente de la escala del flujo (normalizado por el flujo observado)
                - No depende de los errores observacionales, solo de la calidad del ajuste
                - Unidades: porcentaje (%)
                """)
        
        # Guardar features
        if save_results:
            features = extract_features(mcmc_results, phase, flux, flux_err,
                                      sn_name, filter_name)
            features['sn_type'] = selected_type
            
            features_df = pd.DataFrame([features])
            output_file = FEATURES_DIR / f"features_{selected_type.replace(' ', '_')}.csv"
            
            if output_file.exists():
                existing_df = pd.read_csv(output_file)
                mask = (existing_df['sn_name'] == sn_name) & (existing_df['filter_band'] == filter_name)
                if mask.any():
                    existing_df = existing_df[~mask]
                    st.warning(f"Reemplazando entrada existente para {sn_name} - {filter_name}")
                
                combined_df = pd.concat([existing_df, features_df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
            else:
                features_df.to_csv(output_file, index=False)
            
            st.success(f"Features guardadas en: {output_file}")
            
    except Exception as e:
        st.error(f"Error procesando filtro {filter_name}: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

# Sidebar para selección
st.sidebar.header("Configuración")

# Seleccionar tipo de supernova
sn_types = sorted([d.name for d in BASE_DATA_PATH.iterdir() if d.is_dir()], key=str.lower)
selected_type = st.sidebar.selectbox("Tipo de Supernova", sn_types)

# Listar archivos disponibles
type_path = BASE_DATA_PATH / selected_type
dat_files = list(type_path.glob("*_photometry.dat"))

if not dat_files:
    st.error(f"No se encontraron archivos en '{selected_type}'")
    st.stop()

# Seleccionar archivo (ordenado alfabéticamente)
file_options = sorted([f.name for f in dat_files], key=str.lower)
selected_file = st.sidebar.selectbox("Archivo", file_options)
filepath = type_path / selected_file

# Configuración MCMC (usar valores por defecto de config.py)
from config import MCMC_CONFIG

# Botón para procesar (AL PRINCIPIO)
st.sidebar.markdown("---")
st.sidebar.markdown("### Ejecutar Análisis")
if st.sidebar.button("Ejecutar MCMC", type="primary"):
    st.session_state['process'] = True
else:
    st.session_state['process'] = False

st.sidebar.markdown("---")

# Parámetros MCMC
st.sidebar.subheader("Parámetros MCMC")

# Número de walkers
st.sidebar.markdown("**Número de walkers**")
st.sidebar.caption("Cantidad de cadenas MCMC paralelas. Más walkers = mejor exploración del espacio de parámetros pero más lento.")
st.sidebar.caption(f"Valor por defecto (main.py): {MCMC_CONFIG['n_walkers']}")
n_walkers = st.sidebar.slider("Walkers", 20, 100, MCMC_CONFIG["n_walkers"], key="n_walkers")

# Pasos MCMC
st.sidebar.markdown("**Pasos MCMC**")
st.sidebar.caption("Número de iteraciones que cada walker realiza. Más pasos = mejor convergencia pero más tiempo de cómputo.")
st.sidebar.caption(f"Valor por defecto (main.py): {MCMC_CONFIG['n_steps']}")
n_steps = st.sidebar.slider("Pasos", 500, 5000, MCMC_CONFIG["n_steps"], key="n_steps")

# Burn-in
st.sidebar.markdown("**Burn-in**")
st.sidebar.caption("Pasos iniciales a descartar antes de calcular estadísticas. Elimina el período de 'calentamiento' de las cadenas.")
st.sidebar.caption(f"Valor por defecto (main.py): {MCMC_CONFIG['burn_in']}")
burn_in = st.sidebar.slider("Burn-in", 100, 1000, MCMC_CONFIG["burn_in"], key="burn_in")

# Recalcular total_samples con los valores actuales
total_samples = n_walkers * (n_steps - burn_in)
st.sidebar.info(f"**Samples totales**: {total_samples:,} (después de burn-in)")

st.sidebar.markdown("---")
st.sidebar.caption("**Nota**: Estos valores se usan cuando ejecutas `main.py` para procesar múltiples supernovas en batch.")

# Opciones de visualización ANTES de ejecutar
st.sidebar.subheader("Visualización")
st.sidebar.markdown("**Realizaciones MCMC a mostrar**")
st.sidebar.caption("Número de curvas del MCMC a mostrar (0 = solo mediana y promedio). Más curvas = mejor visualización de incertidumbre pero más lento.")
st.sidebar.caption(f"Valor por defecto: 100")
n_samples_to_show = st.sidebar.slider("Realizaciones", 0, 1000, 100, key="n_samples")

st.sidebar.markdown("---")

# Información sobre los gráficos (DESPUÉS de definir todos los parámetros)
st.sidebar.subheader("Información sobre los Gráficos")

st.sidebar.markdown("**Mediana y Promedio del Fit**")
st.sidebar.caption(f"""
**Distribución de parámetros**: El MCMC genera {total_samples:,} **samples**, cada uno es un vector de 6 parámetros: [A, f, t0, t_rise, t_fall, gamma].

Esto significa que tenemos:
- {total_samples:,} valores de A (distribución de A)
- {total_samples:,} valores de f (distribución de f)
- {total_samples:,} valores de t0 (distribución de t0)
- {total_samples:,} valores de t_rise (distribución de t_rise)
- {total_samples:,} valores de t_fall (distribución de t_fall)
- {total_samples:,} valores de gamma (distribución de gamma)

**Mediana (línea azul sólida)**: 
1. Se calcula la **mediana de cada parámetro** por separado:
   - mediana_A = mediana de los {total_samples:,} valores de A
   - mediana_f = mediana de los {total_samples:,} valores de f
   - mediana_t0 = mediana de los {total_samples:,} valores de t0
   - etc.
2. Se obtiene un **vector de parámetros medianos**: [mediana_A, mediana_f, mediana_t0, mediana_t_rise, mediana_t_fall, mediana_gamma]
3. Se evalúa el modelo **UNA VEZ** con ese vector: `alerce_model(phase, mediana_A, mediana_f, ...)`
4. Resultado: **1 curva de luz** (la mediana)

**Promedio (línea verde punteada)**: 
1. Se calcula el **promedio de cada parámetro** por separado:
   - promedio_A = promedio de los {total_samples:,} valores de A
   - promedio_f = promedio de los {total_samples:,} valores de f
   - etc.
2. Se obtiene un **vector de parámetros promedio**: [promedio_A, promedio_f, ...]
3. Se evalúa el modelo **UNA VEZ** con ese vector
4. Resultado: **1 curva de luz** (el promedio)

**IMPORTANTE**: NO se evalúan {total_samples:,} curvas y luego se promedian. Se promedian los **parámetros primero**, luego se evalúa el modelo **1 vez**.

**Diferencia**: La mediana es más robusta a outliers, mientras que el promedio puede verse afectado por valores extremos. Si ambas líneas son similares, la distribución es simétrica.
""")

st.sidebar.markdown("**Red Lines (MCMC Realizations)**")
st.sidebar.caption(f"""
Las líneas rojas semitransparentes son **curvas de luz** generadas evaluando el modelo con diferentes conjuntos de parámetros del MCMC.

**Proceso**:
1. Se seleccionan {n_samples_to_show} conjuntos de parámetros de entre los {total_samples:,} samples disponibles
2. Se evalúa el modelo con cada uno de esos {n_samples_to_show} conjuntos
3. Resultado: **{n_samples_to_show} curvas de luz** (las líneas rojas)

**Selección mejorada**: 
- **20%** de los samples son los más cercanos a la mediana (centro de la distribución)
- **10%** son cercanos al promedio
- El resto se distribuyen en anillos alrededor de la mediana, priorizando cercanía al centro

Esta estrategia asegura que las curvas rojas sean consistentes con las líneas verde (promedio) y azul (mediana), mostrando la incertidumbre de manera representativa.
""")

st.sidebar.markdown("**Corner Plot**")
st.sidebar.caption(f"""
El corner plot muestra la distribución de los **parámetros** (no las curvas) de TODOS los {total_samples:,} samples del MCMC (después de burn-in).

Cada panel muestra:
- **Diagonal**: Histograma de la distribución marginal de cada parámetro (A, f, t0, t_rise, t_fall, gamma)
- **Fuera de la diagonal**: Distribuciones conjuntas de pares de parámetros con contornos de confianza

Esto permite ver correlaciones entre parámetros y la forma de las distribuciones posteriores de los parámetros del modelo.
""")

st.sidebar.markdown("---")

# Opción para guardar resultados ANTES de ejecutar
st.sidebar.subheader("Guardar Resultados")
st.sidebar.markdown("**Guardar gráficos y features**")
st.sidebar.caption("Si está activado, guarda automáticamente los gráficos y features después del cálculo.")
save_results = st.sidebar.checkbox("Guardar gráficos y features", value=False, key="save_results")

st.sidebar.markdown("---")

# Opción para semilla aleatoria (reproducibilidad)
st.sidebar.subheader("Reproducibilidad")
st.sidebar.markdown("**Semilla aleatoria**")
st.sidebar.caption("Fijar semilla para obtener resultados reproducibles. Si cambias los parámetros MCMC, los resultados serán consistentes.")
use_fixed_seed = st.sidebar.checkbox("Usar semilla fija (reproducible)", value=True, key="use_seed")
if use_fixed_seed:
    random_seed = st.sidebar.number_input("Valor de semilla", min_value=0, max_value=999999, value=42, key="seed_value")
else:
    random_seed = None

# Actualizar configuración
MCMC_CONFIG["n_walkers"] = n_walkers
MCMC_CONFIG["n_steps"] = n_steps
MCMC_CONFIG["burn_in"] = burn_in
MCMC_CONFIG["random_seed"] = random_seed

# Contenido principal
if st.session_state.get('process', False):
    with st.spinner("Procesando supernova..."):
        try:
            # Leer archivo
            filters_data, sn_name = parse_photometry_file(str(filepath))
            
            if not filters_data:
                st.error("No se pudieron extraer datos del archivo")
                st.stop()
            
            st.success(f"Supernova: **{sn_name}**")
            st.caption(f"📁 Archivo: `{filepath}`")
            
            # Seleccionar filtros con checkboxes
            available_filters = list(filters_data.keys())
            st.write("**Seleccionar filtros para procesar:**")
            
            # Crear checkboxes en columnas
            n_cols = min(4, len(available_filters))
            cols = st.columns(n_cols)
            selected_filters = []
            
            for i, filter_name in enumerate(available_filters):
                with cols[i % n_cols]:
                    if st.checkbox(filter_name, value=True, key=f"filter_{filter_name}"):
                        selected_filters.append(filter_name)
            
            if not selected_filters:
                st.warning("Debes seleccionar al menos un filtro para procesar")
                st.stop()
            
            st.info(f"Se procesarán **{len(selected_filters)}** curva(s) de luz: {', '.join(selected_filters)}")
            
            # Verificar si hay datos en CSV para mostrar tab de distribuciones
            output_file = FEATURES_DIR / f"features_{selected_type.replace(' ', '_')}.csv"
            has_features_data = False
            if output_file.exists():
                try:
                    df_check = pd.read_csv(output_file)
                    has_features_data = len(df_check) > 0
                except:
                    has_features_data = False
            
            # Siempre usar tabs si hay datos de features para mostrar distribuciones, o si hay múltiples filtros
            if len(selected_filters) == 1 and not has_features_data:
                # Un solo filtro y sin datos previos: mostrar directamente
                filter_name = selected_filters[0]
                _process_single_filter(filters_data, sn_name, filter_name, selected_type, 
                                     n_walkers, n_steps, burn_in, n_samples_to_show, save_results)
            else:
                # Múltiples filtros o hay datos para distribuciones: usar tabs
                tab_names = [f"Filter {f}" for f in selected_filters]
                if has_features_data:
                    tab_names.append("Distributions")
                
                tabs = st.tabs(tab_names)
                
                # Tabs para cada filtro
                for tab, filter_name in zip(tabs[:len(selected_filters)], selected_filters):
                    with tab:
                        _process_single_filter(filters_data, sn_name, filter_name, selected_type,
                                             n_walkers, n_steps, burn_in, n_samples_to_show, save_results)
                
                # Tab de distribuciones de parámetros (si hay datos)
                if has_features_data:
                    with tabs[-1]:
                        _show_parameter_distributions(selected_type)
            
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
else:
    # Mostrar distribuciones si hay datos disponibles, sin necesidad de ejecutar MCMC
    output_file = FEATURES_DIR / f"features_{selected_type.replace(' ', '_')}.csv"
    has_features_data = False
    if output_file.exists():
        try:
            df_check = pd.read_csv(output_file)
            has_features_data = len(df_check) > 0
        except:
            has_features_data = False
    
    if has_features_data:
        st.info("**Tip**: Puedes ver las distribuciones de parámetros sin ejecutar MCMC. Las distribuciones se muestran abajo basadas en los datos ya procesados.")
        st.markdown("---")
        _show_parameter_distributions(selected_type)
    else:
        st.info("Selecciona una supernova y configura los parámetros MCMC en la barra lateral, luego presiona 'Ejecutar MCMC'")

