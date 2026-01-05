"""
Script principal para extraer features de supernovas con MCMC

Uso:
    python main.py "SN Ia" 3
    python main.py "SN Ia" 3 "g,r"  # Con filtros específicos
    python main.py "SN Ia" 3 --resume  # Continuar desde checkpoint
"""
import sys
import time
import json
import logging
import gc
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from reader import parse_photometry_file, prepare_lightcurve, load_supernovas_from_csv
from mcmc_fitter import fit_mcmc, validate_physical_fit
from feature_extractor import extract_features
from plotter import plot_corner, plot_fit_with_uncertainty, plot_extended_model
from config import BASE_DATA_PATH, PLOTS_DIR, FEATURES_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, DEBUG_PDF_DIR, FILTERS_TO_PROCESS, MCMC_CONFIG, DATA_FILTER_CONFIG, PLOT_CONFIG

# Configurar logging
def setup_logger(sn_type):
    """
    Configurar logger para esta ejecución
    
    Returns:
    --------
    logger : logging.Logger
    """
    log_file = LOG_DIR / f"log_{sn_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(f"mcmc_extraction_{sn_type}")
    logger.setLevel(logging.INFO)
    
    # Evitar duplicados de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"="*80)
    logger.info(f"Iniciando extracción de features para: {sn_type}")
    logger.info(f"Archivo de log: {log_file}")
    logger.info(f"="*80)
    
    return logger

def process_single_filter(filters_data, sn_name, filter_name, sn_type, logger=None):
    """
    Procesar un filtro específico de una supernova
    
    Parameters:
    -----------
    filters_data : dict
        Datos de todos los filtros
    sn_name : str
        Nombre de la supernova
    filter_name : str
        Nombre del filtro a procesar
    sn_type : str
        Tipo de supernova
    logger : logging.Logger, optional
        Logger para registrar eventos
        
    Returns:
    --------
    dict con features o None si falla
    """
    print(f"\n  --- Procesando filtro: {filter_name} ---")
    if logger:
        logger.info(f"  [{sn_name}] Iniciando procesamiento del filtro: {filter_name}")
    
    try:
        # Preparar datos con filtro temporal (solo hasta 300 días después del pico)
        if logger:
            logger.info(f"  [{sn_name} | {filter_name}] Paso 1/5: Preparando datos de curva de luz")
        
        lc_data = prepare_lightcurve(
            filters_data[filter_name], 
            filter_name,
            max_days_after_peak=DATA_FILTER_CONFIG["max_days_after_peak"],
            max_days_before_peak=DATA_FILTER_CONFIG["max_days_before_peak"],
            max_days_before_first_obs=DATA_FILTER_CONFIG["max_days_before_first_obs"]
        )
        
        if lc_data is None:
            error_msg = f"No hay suficientes datos para filtro {filter_name}"
            print(f"    [ERROR] {error_msg}")
            if logger:
                logger.error(f"  [{sn_name} | {filter_name}] ERROR en Paso 1/5: {error_msg}")
            return None
        
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
        
        # Contar puntos normales vs totales (incluyendo upper limits)
        n_normal = np.sum(~is_upper_limit) if is_upper_limit is not None else len(phase)
        n_total = len(phase)
        
        print(f"    [OK] Puntos de datos para MCMC: {n_total} ({n_normal} detecciones normales", end="")
        if is_upper_limit is not None and np.any(is_upper_limit):
            n_ul = np.sum(is_upper_limit)
            print(f" + {n_ul} upper limits)", end="")
        else:
            print(")", end="")
        print()
        
        # Mostrar rango en MJD si está disponible, sino en fase
        if mjd is not None and len(mjd) > 0:
            print(f"    [OK] Rango MJD: {mjd.min():.1f} - {mjd.max():.1f}")
            if peak_phase is not None and reference_mjd is not None:
                peak_mjd = reference_mjd + peak_phase
                print(f"    [OK] Peak en MJD: {peak_mjd:.1f} (fase relativa: {peak_phase:.1f} días)")
        else:
            print(f"    [OK] Rango de fase: {phase.min():.1f} - {phase.max():.1f} días")
            if peak_phase is not None:
                print(f"    [OK] Peak phase: {peak_phase:.1f} días")
        
        # Construir texto del filtro temporal
        if DATA_FILTER_CONFIG['max_days_before_peak'] is None:
            filter_text = f"Solo datos hasta {DATA_FILTER_CONFIG['max_days_after_peak']:.0f} días después del peak (sin límite antes del peak)"
        else:
            filter_text = f"{DATA_FILTER_CONFIG['max_days_before_peak']:.0f} días antes y {DATA_FILTER_CONFIG['max_days_after_peak']:.0f} días después del peak"
        print(f"    [OK] Datos filtrados: {filter_text}")
        
        # Ajuste MCMC
        if logger:
            logger.info(f"  [{sn_name} | {filter_name}] Paso 2/5: Ejecutando MCMC (walkers={MCMC_CONFIG['n_walkers']}, steps={MCMC_CONFIG['n_steps']})")
        print(f"    Ejecutando MCMC...")
        t0_mcmc = time.time()
        
        try:
            mcmc_results = fit_mcmc(phase, flux, flux_err, verbose=False, is_upper_limit=is_upper_limit)
            t_mcmc = time.time() - t0_mcmc
            print(f"    [OK] MCMC completado en {t_mcmc:.2f} segundos")
            if logger:
                logger.info(f"  [{sn_name} | {filter_name}] Paso 2/5: MCMC completado exitosamente en {t_mcmc:.2f}s")
            
            # Validar comportamiento físico del fit
            is_valid, reason = validate_physical_fit(mcmc_results, phase, flux, is_upper_limit)
            if not is_valid:
                error_msg = f"Fit no físico: {reason}"
                print(f"    [REJECT] {error_msg}")
                if logger:
                    logger.warning(f"  [{sn_name} | {filter_name}] Fit rechazado: {reason}")
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error en MCMC: {type(e).__name__}: {str(e)}"
            print(f"    [ERROR] {error_msg}")
            if logger:
                logger.error(f"  [{sn_name} | {filter_name}] ERROR en Paso 2/5: {error_msg}")
            raise
        
        # Convertir flujo del modelo a magnitud
        from model import flux_to_mag
        mag_model = flux_to_mag(np.clip(mcmc_results['model_flux'], 1e-10, None))
        
        # Extraer features
        if logger:
            logger.info(f"  [{sn_name} | {filter_name}] Paso 3/5: Extrayendo features")
        t0_features = time.time()
        
        try:
            features = extract_features(mcmc_results, phase, flux, flux_err,
                                       sn_name, filter_name)
            features['sn_type'] = sn_type
            t_features = time.time() - t0_features
            print(f"    [OK] Features extraídas en {t_features:.3f} segundos")
            if logger:
                logger.info(f"  [{sn_name} | {filter_name}] Paso 3/5: Features extraídas exitosamente en {t_features:.3f}s")
        except Exception as e:
            error_msg = f"Error extrayendo features: {type(e).__name__}: {str(e)}"
            print(f"    [ERROR] {error_msg}")
            if logger:
                logger.error(f"  [{sn_name} | {filter_name}] ERROR en Paso 3/5: {error_msg}")
            raise
        
        # Convertir modelo a MJD para plotting (igual que en modo debug)
        # Usar samples_valid si está disponible (samples que respetan upper limits)
        # para calcular mediana/promedio, pero todos los samples para visualización
        samples_to_use = mcmc_results.get('samples_valid', mcmc_results['samples'])
        
        if mjd is not None and reference_mjd is not None:
            # Ajustar samples: t0 está en fase relativa, convertirlo a MJD absoluto
            from model import alerce_model
            samples_mjd = samples_to_use.copy()
            samples_mjd[:, 2] = samples_mjd[:, 2] + reference_mjd  # t0 en MJD absoluto
            
            # Usar EXACTAMENTE mcmc_results['params'] (la misma mediana que el corner plot)
            # Solo convertir t0 a MJD para el eje X
            param_medians_mjd = mcmc_results['params'].copy()
            param_medians_mjd[2] = param_medians_mjd[2] + reference_mjd  # t0 a MJD
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
            samples_for_plot = samples_to_use
        
        # Crear subcarpeta para esta supernova solo si vamos a guardar algo
        # Organizar por tipo de supernova: plots/SN Ia/ZTF20abc/
        sn_plots_dir = PLOTS_DIR / sn_type / sn_name
        sn_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar gráficos (con múltiples realizaciones MCMC)
        plot_filename = f"{sn_name}_{filter_name}_fit.png"
        plot_path = sn_plots_dir / plot_filename
        t0_plot = time.time()
        # Calcular número total de samples usados para la mediana
        n_total_samples = len(mcmc_results['samples'])
        print(f"    [INFO] Samples totales para mediana: {n_total_samples:,} (de {MCMC_CONFIG['n_walkers']} walkers × {MCMC_CONFIG['n_steps'] - MCMC_CONFIG['burn_in']} pasos)")
        # plot_fit_with_uncertainty ahora retorna (fig, sample_indices, central_curve_sample_idx)
        # Usar param_medians_mjd si existe (cuando el eje X está en MJD)
        # También convertir params_median_of_curves a MJD para consistencia
        if reference_mjd is not None and len(mjd) > 0:
            param_medians_for_plot = param_medians_mjd
            # Convertir params_median_of_curves a MJD
            params_moc = mcmc_results.get('params_median_of_curves', None)
            if params_moc is not None:
                params_moc_mjd = params_moc.copy()
                params_moc_mjd[2] = params_moc_mjd[2] + reference_mjd  # t0 a MJD
            else:
                params_moc_mjd = None
        else:
            param_medians_for_plot = mcmc_results['params']
            params_moc_mjd = mcmc_results.get('params_median_of_curves', None)
        
        _, sample_indices, central_curve_idx = plot_fit_with_uncertainty(
            phase_for_plot, mag, mag_err, mag_model_for_plot, flux, flux_model_for_plot,
            samples_for_plot, n_samples_to_show=100,  # Valor por defecto: 100 realizaciones para visualización
            sn_name=sn_name, filter_name=filter_name, save_path=str(plot_path),
            is_upper_limit=is_upper_limit, flux_err=flux_err,
            had_upper_limits=had_upper_limits,
            param_medians_phase_relative=mcmc_results['params'],  # Para debug
            param_medians=param_medians_for_plot,  # Mediana real para la línea azul
            params_median_of_curves=params_moc_mjd,  # Curva central (verde)
            dynamic_bounds=mcmc_results.get('dynamic_bounds', None)  # Bounds dinámicos del ajuste
        )
        t_plot = time.time() - t0_plot
        print(f"    [OK] Gráfico guardado en {t_plot:.2f} segundos: {plot_path}")
        
        # Modelo extendido (figura separada)
        # Usamos el mismo sample de la curva central que en el plot normal
        from plotter import plot_extended_model
        extended_filename = f"{sn_name}_{filter_name}_extended.png"
        extended_path = sn_plots_dir / extended_filename
        t0_extended = time.time()
        plot_extended_model(
            phase_for_plot, flux, mcmc_results['params'],
            is_upper_limit=is_upper_limit,
            flux_err=flux_err,
            sn_name=sn_name, filter_name=filter_name, save_path=str(extended_path),
            samples=samples_for_plot,
            precalculated_sample_indices=sample_indices,
            central_curve_sample_idx=central_curve_idx  # Mismo sample que en plot normal
        )
        t_extended = time.time() - t0_extended
        print(f"    [OK] Modelo extendido guardado en {t_extended:.2f} segundos: {extended_path}")
        
        # Corner plot
        # IMPORTANTE: Usar samples originales en fase relativa (no convertidos a MJD)
        # Los parámetros deben estar en fase relativa para consistencia entre supernovas
        corner_filename = f"{sn_name}_{filter_name}_corner.png"
        corner_path = sn_plots_dir / corner_filename
        t0_corner = time.time()
        # Usar las 200 mejores curvas para el corner plot (consistente con param_medians)
        samples_for_corner = mcmc_results.get('samples_best_200', mcmc_results.get('samples_valid', mcmc_results['samples']))
        plot_corner(samples_for_corner, save_path=str(corner_path), 
                   param_medians=mcmc_results['params'],
                   param_percentiles=mcmc_results['params_percentiles'])
        t_corner = time.time() - t0_corner
        print(f"    [OK] Corner plot guardado en {t_corner:.2f} segundos: {corner_path}")
        
        # Liberar memoria: eliminar samples grandes del MCMC después de usarlos
        del mcmc_results['samples']
        gc.collect()  # Forzar recolección de basura
        
        t_total_filter = t_mcmc + t_features + t_plot + t_corner
        print(f"    [OK] Tiempo total para filtro {filter_name}: {t_total_filter:.2f} segundos")
        print(f"    [OK] Features extraídas para filtro {filter_name}")
        if logger:
            logger.info(f"  [{sn_name} | {filter_name}] COMPLETADO exitosamente en {t_total_filter:.2f}s")
        return features
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"    [ERROR] {error_msg}")
        if logger:
            logger.error(f"  [{sn_name} | {filter_name}] FALLO: {error_msg}")
            import traceback
            logger.error(f"  [{sn_name} | {filter_name}] Traceback:\n{traceback.format_exc()}")
        import traceback
        traceback.print_exc()
        return None

def load_checkpoint(sn_type):
    """
    Cargar checkpoint de supernovas ya procesadas
    
    Returns:
    --------
    set : Conjunto de tuplas (sn_name, filter_name) ya procesadas
    """
    checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{sn_type.replace(' ', '_')}.json"
    
    if not checkpoint_file.exists():
        return set()
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            # Convertir lista de listas a set de tuplas
            processed = set(tuple(item) for item in data.get('processed', []))
            return processed
    except Exception as e:
        print(f"  [WARNING] Error al leer checkpoint: {e}")
        return set()

def save_checkpoint(sn_type, processed_set):
    """
    Guardar checkpoint de supernovas procesadas
    
    Parameters:
    -----------
    sn_type : str
        Tipo de supernova
    processed_set : set
        Conjunto de tuplas (sn_name, filter_name) procesadas
    """
    checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{sn_type.replace(' ', '_')}.json"
    
    try:
        # Convertir set de tuplas a lista de listas para JSON
        data = {
            'sn_type': sn_type,
            'processed': [list(item) for item in processed_set],
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [WARNING] Error al guardar checkpoint: {e}")

def load_debug_checkpoint(sn_type):
    """
    Cargar checkpoint de supernovas ya procesadas en modo debug
    
    Returns:
    --------
    set : Conjunto de nombres de supernovas ya procesadas
    """
    checkpoint_file = CHECKPOINT_DIR / f"debug_checkpoint_{sn_type.replace(' ', '_')}.json"
    
    if not checkpoint_file.exists():
        return set()
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            processed = set(data.get('processed', []))
            return processed
    except Exception as e:
        print(f"  [WARNING] Error al leer checkpoint de debug: {e}")
        return set()

def save_debug_checkpoint(sn_type, processed_set):
    """
    Guardar checkpoint de supernovas procesadas en modo debug
    
    Parameters:
    -----------
    sn_type : str
        Tipo de supernova
    processed_set : set
        Conjunto de nombres de supernovas procesadas
    """
    checkpoint_file = CHECKPOINT_DIR / f"debug_checkpoint_{sn_type.replace(' ', '_')}.json"
    
    try:
        data = {
            'sn_type': sn_type,
            'processed': list(processed_set),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [WARNING] Error al guardar checkpoint de debug: {e}")

def save_features_incremental(features_list, sn_type, debug_mode=False, from_csv=False, overwrite_pdf=False):
    """
    Guardar features incrementalmente en CSV después de procesar una supernova.
    Esto asegura que si el proceso se cae, las features ya procesadas estén guardadas.
    
    Parameters:
    -----------
    features_list : list
        Lista de dicts con features (uno por filtro procesado)
    sn_type : str
        Tipo de supernova
    debug_mode : bool
        Si True, guarda en archivo separado para modo debug
    from_csv : bool
        Si True y debug_mode=True, agrega sufijo _from_csv al nombre
    overwrite_pdf : bool
        Si True y from_csv=True, usa el mismo nombre que el modo normal (sobrescribir)
    """
    if not features_list:
        return
    
    try:
        # Determinar nombre del archivo según el modo
        if debug_mode:
            if from_csv:
                # Si se usa CSV, siempre usar sufijo _from_csv (independiente de --overwrite)
                output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}_debug_from_csv.csv"
            else:
                output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}_debug.csv"
        else:
            output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}.csv"
        
        df_new_features = pd.DataFrame(features_list)
        
        # Si el archivo ya existe, leerlo y actualizar
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            
            # Eliminar entradas existentes para las mismas combinaciones (sn_name, filter_band)
            for _, row in df_new_features.iterrows():
                mask = (existing_df['sn_name'] == row['sn_name']) & (existing_df['filter_band'] == row['filter_band'])
                if mask.any():
                    existing_df = existing_df[~mask]
            
            # Combinar: primero las existentes (sin duplicados), luego las nuevas
            combined_df = pd.concat([existing_df, df_new_features], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
        else:
            # Si no existe, crear nuevo archivo
            df_new_features.to_csv(output_file, index=False)
    except Exception as e:
        print(f"  [WARNING] Error al guardar features incrementalmente: {e}")

def is_already_processed(sn_name, filter_name, processed_set):
    """
    Verificar si una combinación sn_name + filter ya fue procesada
    """
    return (sn_name, filter_name) in processed_set

def process_supernova(filepath, sn_type, filters_to_process=None, processed_set=None, logger=None):
    """
    Procesar una supernova completa (múltiples filtros)
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo .dat
    sn_type : str
        Tipo de supernova
    filters_to_process : list, optional
        Lista de filtros a procesar (ej: ['g', 'r'])
        Si es None, usa FILTERS_TO_PROCESS de config.py
        Si está vacía, procesa todos los filtros disponibles
        
    Returns:
    --------
    list de dicts con features (uno por filtro procesado)
    """
    print(f"\n{'='*80}")
    print(f"Procesando: {Path(filepath).name}")
    print(f"{'='*80}")
    
    sn_name = None
    try:
        # Leer archivo
        if logger:
            logger.info("")
            logger.info("-" * 80)
            logger.info(f"SUPERNOVA: {Path(filepath).name}")
            logger.info("-" * 80)
            logger.info(f"[{Path(filepath).name}] Iniciando procesamiento de supernova")
        filters_data, sn_name = parse_photometry_file(filepath)
        
        if not filters_data:
            error_msg = "No se pudieron extraer datos del archivo"
            print(f"  [ERROR] {error_msg}")
            if logger:
                logger.error(f"[{Path(filepath).name}] ERROR: {error_msg}")
            return []
        
        print(f"  [OK] Supernova: {sn_name}")
        print(f"  [OK] Filtros disponibles: {', '.join(filters_data.keys())}")
        if logger:
            logger.info(f"[{sn_name}] Supernova identificada, filtros disponibles: {', '.join(filters_data.keys())}")
        
        # Determinar qué filtros procesar
        if filters_to_process is None:
            filters_to_process = FILTERS_TO_PROCESS
        
        if not filters_to_process:
            # Si la lista está vacía, procesar todos los filtros disponibles
            filters_to_process = list(filters_data.keys())
            print(f"  [INFO] Procesando todos los filtros disponibles: {', '.join(filters_to_process)}")
        else:
            # Filtrar solo los que existen
            available_filters = [f for f in filters_to_process if f in filters_data]
            missing_filters = [f for f in filters_to_process if f not in filters_data]
            
            if missing_filters:
                print(f"  [WARNING] Filtros no disponibles: {', '.join(missing_filters)}")
            
            if not available_filters:
                print(f"  [ERROR] Ninguno de los filtros solicitados está disponible")
                return []
            
            filters_to_process = available_filters
            print(f"  [INFO] Procesando filtros: {', '.join(filters_to_process)}")
        
        # Procesar cada filtro
        all_features = []
        skipped_count = 0
        t0_sn = time.time()
        
        for filter_name in filters_to_process:
            # Verificar si ya fue procesado (si hay checkpoint)
            if processed_set is not None:
                if is_already_processed(sn_name, filter_name, processed_set):
                    skip_msg = f"{sn_name} - {filter_name} ya procesado (checkpoint)"
                    print(f"  [SKIP] {skip_msg}")
                    if logger:
                        logger.info(f"  [{sn_name} | {filter_name}] SKIP: Ya procesado en checkpoint anterior")
                    skipped_count += 1
                    continue
            
            features = process_single_filter(filters_data, sn_name, filter_name, sn_type, logger)
            if features:
                all_features.append(features)
                # Actualizar checkpoint inmediatamente después de procesar exitosamente
                if processed_set is not None:
                    processed_set.add((sn_name, filter_name))
                    save_checkpoint(sn_type, processed_set)
        
        # Liberar memoria después de procesar todos los filtros de una supernova
        del filters_data
        import gc
        gc.collect()
        
        # Eliminar carpeta si está vacía (no se procesó ningún filtro exitosamente)
        if sn_name:
            sn_plots_dir = PLOTS_DIR / sn_type / sn_name
            if sn_plots_dir.exists() and sn_plots_dir.is_dir():
                # Verificar si la carpeta está vacía
                try:
                    if not any(sn_plots_dir.iterdir()):
                        sn_plots_dir.rmdir()
                        print(f"  [INFO] Carpeta vacía eliminada: {sn_plots_dir}")
                except OSError:
                    pass  # Si no se puede eliminar, no es crítico
        
        t_total_sn = time.time() - t0_sn
        
        # Construir mensaje de resumen
        if skipped_count > 0:
            if len(all_features) > 0:
                success_msg = f"Procesados {len(all_features)}/{len(filters_to_process)} filtros exitosamente ({skipped_count} saltados por checkpoint)"
            else:
                success_msg = f"Todos los filtros ya estaban procesados ({skipped_count}/{len(filters_to_process)} saltados por checkpoint)"
        else:
            success_msg = f"Procesados {len(all_features)}/{len(filters_to_process)} filtros exitosamente"
        
        print(f"\n  [OK] {success_msg}")
        print(f"  [OK] Tiempo total para {sn_name}: {t_total_sn:.2f} segundos ({t_total_sn/60:.2f} minutos)")
        if logger:
            logger.info(f"[{sn_name}] COMPLETADO: {success_msg} en {t_total_sn:.2f}s")
        return all_features
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"  [ERROR] {error_msg}")
        if logger and sn_name:
            logger.error(f"[{sn_name}] ERROR GENERAL: {error_msg}")
            import traceback
            logger.error(f"[{sn_name}] Traceback:\n{traceback.format_exc()}")
        elif logger:
            logger.error(f"[{Path(filepath).name}] ERROR GENERAL: {error_msg}")
        import traceback
        traceback.print_exc()
        return []


def filter_by_year(dat_files, min_year=2022):
    """
    Filtrar archivos por año (basado en el nombre ZTF)
    
    Parameters:
    -----------
    dat_files : list
        Lista de Paths a archivos .dat
    min_year : int
        Año mínimo (default: 2022)
        
    Returns:
    --------
    filtered_files : list
        Archivos filtrados, ordenados por año (más recientes primero)
    """
    filtered = []
    for filepath in dat_files:
        filename = filepath.name
        # Extraer año del nombre: ZTF22, ZTF23, etc.
        if filename.startswith('ZTF'):
            try:
                year_str = filename[3:5]  # ZTF22 -> 22
                year = 2000 + int(year_str)
                if year >= min_year:
                    filtered.append((year, filepath))
            except (ValueError, IndexError):
                continue
    
    # Ordenar por año descendente (más recientes primero)
    filtered.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in filtered]


def _save_page_to_pdf(fig, pdf_path, pdf_exists=False):
    """
    Guardar una página al PDF, abriendo y cerrando inmediatamente para evitar acumular en memoria.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figura a guardar
    pdf_path : Path
        Ruta al archivo PDF
    pdf_exists : bool
        Si True y el PDF existe, añade la página usando PyPDF2. Si False, crea nuevo PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import traceback
    
    print(f"  [DEBUG] _save_page_to_pdf llamado:")
    print(f"    - pdf_path: {pdf_path}")
    print(f"    - pdf_exists: {pdf_exists}")
    print(f"    - pdf_path.exists(): {pdf_path.exists()}")
    print(f"    - fig es None: {fig is None}")
    
    if fig is None:
        print(f"  [ERROR] La figura es None, no se puede guardar")
        return
    
    try:
        from PyPDF2 import PdfReader, PdfWriter
        HAS_PYPDF2 = True
        print(f"    - PyPDF2 disponible: True")
    except ImportError:
        HAS_PYPDF2 = False
        print(f"    - PyPDF2 disponible: False")
    
    # Si el PDF existe y tenemos PyPDF2, usar append mode
    if pdf_exists and pdf_path.exists() and HAS_PYPDF2:
        # Modo append: leer PDF original, añadir nueva página, guardar
        print(f"  [DEBUG] Modo append: añadiendo página al PDF existente")
        try:
            # Guardar nueva página en PDF temporal
            temp_pdf = pdf_path.parent / f"{pdf_path.stem}_temp_page.pdf"
            print(f"  [DEBUG] Guardando página temporal en: {temp_pdf}")
            with PdfPages(str(temp_pdf)) as temp_pdf_file:
                temp_pdf_file.savefig(fig, bbox_inches='tight', dpi=200)
            print(f"  [DEBUG] Página temporal guardada exitosamente")
            
            # Leer PDF original y PDF temporal
            reader_original = PdfReader(str(pdf_path))
            reader_temp = PdfReader(str(temp_pdf))
            
            # Crear writer para PDF combinado
            writer = PdfWriter()
            
            # Añadir todas las páginas del PDF original
            for page in reader_original.pages:
                writer.add_page(page)
            
            # Añadir la nueva página
            writer.add_page(reader_temp.pages[0])
            
            # Guardar PDF combinado
            print(f"  [DEBUG] Guardando PDF combinado en: {pdf_path}")
            with open(pdf_path, 'wb') as output_file:
                writer.write(output_file)
            print(f"  [DEBUG] PDF combinado guardado exitosamente")
            
            # Eliminar PDF temporal
            temp_pdf.unlink()
            print(f"  [DEBUG] PDF temporal eliminado")
            
        except Exception as e:
            print(f"  [ERROR] Error al añadir página al PDF existente: {e}")
            print(f"  [ERROR] Traceback:")
            traceback.print_exc()
            print(f"  [INFO] Guardando como nuevo PDF...")
            # Fallback: crear nuevo PDF
            try:
                with PdfPages(str(pdf_path)) as pdf:
                    pdf.savefig(fig, bbox_inches='tight', dpi=200)
                print(f"  [DEBUG] PDF creado exitosamente (fallback)")
            except Exception as e2:
                print(f"  [ERROR] Error al crear PDF (fallback): {e2}")
                traceback.print_exc()
    else:
        # Modo normal: crear nuevo PDF (PdfPages sobrescribe si existe)
        print(f"  [DEBUG] Modo normal: creando nuevo PDF")
        try:
            with PdfPages(str(pdf_path)) as pdf:
                pdf.savefig(fig, bbox_inches='tight', dpi=200)
            print(f"  [DEBUG] PDF creado exitosamente en: {pdf_path}")
            print(f"  [DEBUG] PDF existe después de guardar: {pdf_path.exists()}")
            if pdf_path.exists():
                print(f"  [DEBUG] Tamaño del PDF: {pdf_path.stat().st_size} bytes")
        except Exception as e:
            print(f"  [ERROR] Error al crear PDF: {e}")
            print(f"  [ERROR] Traceback:")
            traceback.print_exc()

def generate_debug_pdf(sn_type, n_supernovas, filters_to_process=None, min_year=2022, 
                       resume_from_checkpoint=False, supernovas_from_csv=None, csv_file_path=None,
                       overwrite_pdf=False):
    """
    Generar PDF de debug con fit + corner plot para múltiples supernovas
    Usa las mismas funciones de plotting que el procesamiento normal
    
    Parameters:
    -----------
    sn_type : str
        Tipo de supernova
    n_supernovas : int or None
        Número de supernovas a procesar (None = todas). Se ignora si supernovas_from_csv está especificado.
    filters_to_process : list, optional
        Lista de filtros a procesar
    min_year : int
        Año mínimo para filtrar supernovas (se ignora si supernovas_from_csv está especificado)
    resume_from_checkpoint : bool
        Si True, reanuda desde checkpoint y añade al PDF existente
    supernovas_from_csv : list, optional
        Lista de nombres de supernovas del CSV a procesar. Si se especifica, se procesan solo estas.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import io
    import matplotlib.image as mpimg
    try:
        from PyPDF2 import PdfReader, PdfWriter
        HAS_PYPDF2 = True
    except ImportError:
        HAS_PYPDF2 = False
        if resume_from_checkpoint:
            print("[WARNING] PyPDF2 no está instalado. No se puede añadir páginas al PDF existente.")
            print("[WARNING] Instala con: pip install PyPDF2")
            print("[WARNING] Continuando sin reanudar desde checkpoint...")
            resume_from_checkpoint = False
    
    print(f"\n{'='*80}")
    print(f"GENERANDO PDF DE DEBUG")
    print(f"{'='*80}")
    print(f"Tipo de supernova: {sn_type}")
    if supernovas_from_csv:
        print(f"Modo CSV: procesando {len(supernovas_from_csv)} supernovas específicas del CSV")
        print(f"(Se ignoran n_supernovas y filtro de año)")
    else:
        if n_supernovas is None:
            print(f"Número de supernovas: TODAS")
        else:
            print(f"Número de supernovas: {n_supernovas}")
        print(f"Año mínimo inicial: {min_year} (se reducirá automáticamente si no hay suficientes)")
    if filters_to_process:
        print(f"Filtros: {', '.join(filters_to_process)}")
    else:
        print(f"Filtros: {FILTERS_TO_PROCESS} (desde config.py)")
    
    # Buscar archivos
    type_path = BASE_DATA_PATH / sn_type
    
    if not type_path.exists():
        print(f"\n[ERROR] La carpeta '{sn_type}' no existe")
        return
    
    dat_files = list(type_path.glob("*_photometry.dat"))
    
    if not dat_files:
        print(f"\n[ERROR] No se encontraron archivos .dat en '{sn_type}'")
        return
    
    # Si se especificó CSV, filtrar archivos para procesar solo esas supernovas
    if supernovas_from_csv:
        # Crear un diccionario para mapear nombres de supernovas a archivos
        # El nombre del archivo es como "ZTF18aaaibml_photometry.dat" y la supernova es "ZTF18aaaibml"
        sn_to_file = {}
        for filepath in dat_files:
            # Extraer nombre de supernova del archivo
            file_stem = Path(filepath).stem.replace('_photometry', '')
            sn_to_file[file_stem] = filepath
        
        # Filtrar archivos para solo incluir las supernovas del CSV
        selected_files = []
        not_found = []
        for sn_name in supernovas_from_csv:
            if sn_name in sn_to_file:
                selected_files.append(sn_to_file[sn_name])
            else:
                not_found.append(sn_name)
        
        if not_found:
            print(f"[WARNING] {len(not_found)} supernovas del CSV no se encontraron en los archivos .dat:")
            for sn in not_found[:10]:  # Mostrar solo las primeras 10
                print(f"  - {sn}")
            if len(not_found) > 10:
                print(f"  ... y {len(not_found) - 10} más")
        
        if not selected_files:
            print(f"\n[ERROR] No se encontraron archivos .dat para ninguna supernova del CSV")
            return
        
        print(f"[INFO] {len(selected_files)} archivos encontrados de {len(supernovas_from_csv)} supernovas del CSV")
        # Usar selected_files como la lista de archivos a procesar
        # Saltar el filtrado por año ya que estamos usando archivos específicos
        filtered_files = selected_files
        # No necesitamos shuffle porque queremos mantener el orden del CSV
        # Pero podemos hacerlo opcional si el usuario quiere
    else:
        # Comportamiento normal: filtrar por año y seleccionar aleatoriamente
        filtered_files = None  # Se asignará más abajo
    
    # Crear nombre del archivo PDF
    if supernovas_from_csv and csv_file_path:
        # Si se usa CSV, siempre usar sufijo _from_csv (independiente de --overwrite)
        pdf_filename = sn_type.replace(' ', '_').replace('-', '_') + '_debug_from_csv.pdf'
        print(f"[INFO] Modo from-csv: usando PDF: {pdf_filename}")
    else:
        pdf_filename = sn_type.replace(' ', '_').replace('-', '_') + '_debug.pdf'
        print(f"[INFO] Modo normal: usando PDF: {pdf_filename}")
    pdf_path = DEBUG_PDF_DIR / pdf_filename
    print(f"[INFO] Ruta completa del PDF: {pdf_path}")
    
    # Si se usa --overwrite, borrar los archivos existentes UNA VEZ al inicio
    if overwrite_pdf:
        # Eliminar PDF existente
        if pdf_path.exists():
            try:
                pdf_path.unlink()
                print(f"[INFO] PDF existente eliminado (modo overwrite)")
            except Exception as e:
                print(f"[WARNING] No se pudo eliminar el PDF existente: {e}")
        
        # Eliminar CSV de features existente (solo en modo debug)
        from config import FEATURES_DIR
        if supernovas_from_csv and csv_file_path:
            # Si se usa CSV, siempre usar sufijo _from_csv (independiente de --overwrite)
            features_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}_debug_from_csv.csv"
        else:
            features_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}_debug.csv"
        
        if features_file.exists():
            try:
                features_file.unlink()
                print(f"[INFO] Archivo de features existente eliminado (modo overwrite)")
            except Exception as e:
                print(f"[WARNING] No se pudo eliminar el archivo de features existente: {e}")
    
    # Cargar checkpoint PRIMERO para saber si estamos reanudando
    processed_supernovas = set()
    pdf_exists = pdf_path.exists()
    has_checkpoint = False
    
    if resume_from_checkpoint:
        processed_supernovas = load_debug_checkpoint(sn_type)
        n_processed = len(processed_supernovas)
        if n_processed > 0:
            has_checkpoint = True
            print(f"[INFO] Checkpoint de debug cargado: {n_processed} supernovas ya procesadas")
            if pdf_exists:
                print(f"[INFO] PDF existente encontrado: {pdf_path}")
                print(f"[INFO] Se añadirán nuevas páginas al PDF existente")
            else:
                print(f"[WARNING] Checkpoint encontrado pero PDF no existe. Creando nuevo PDF.")
        else:
            print(f"[INFO] No se encontró checkpoint de debug, comenzando desde el inicio")
            resume_from_checkpoint = False
    elif pdf_exists:
        if overwrite_pdf:
            print(f"[INFO] PDF existente encontrado: {pdf_path}")
            print(f"[INFO] Modo overwrite activado: se sobrescribirá el PDF existente")
        else:
            print(f"[WARNING] PDF existente encontrado: {pdf_path}")
            print(f"[WARNING] Se sobrescribirá. Usa --resume para añadir páginas al PDF existente, o --overwrite para sobrescribir explícitamente.")
    
    # Filtrar por año solo si NO estamos usando modo CSV
    current_year = None  # Inicializar para evitar errores
    if not supernovas_from_csv:
        # Filtrar por año, reduciendo automáticamente si no hay suficientes
        # PERO: si estamos reanudando desde checkpoint, mantener el filtro de min_year sin reducir
        current_year = min_year
        min_year_limit = 2018  # ZTF empezó alrededor de 2018
        filtered_files_tuples = []
        
        # Si estamos reanudando, no reducir el año - mantener min_year estricto
        if has_checkpoint:
            filtered_files_tuples = filter_by_year(dat_files, min_year=min_year)
            print(f"[INFO] Modo resume: manteniendo filtro de año {min_year} en adelante (sin reducir)")
        else:
            # Modo normal: reducir año si no hay suficientes
            while current_year >= min_year_limit:
                filtered_files_tuples = filter_by_year(dat_files, min_year=current_year)
                n_found = len(filtered_files_tuples)
                n_needed = n_supernovas if n_supernovas is not None else 1
                
                if n_found >= n_needed:
                    if current_year < min_year:
                        print(f"[INFO] No se encontraron suficientes supernovas desde {min_year}, reduciendo a {current_year}")
                    break
                
                # Mostrar progreso si está reduciendo el año
                if current_year < min_year:
                    print(f"[INFO] Probando año {current_year}: {n_found} archivos encontrados (necesarios: {n_needed})")
                
                current_year -= 1
        
        if not filtered_files_tuples:
            print(f"\n[ERROR] No se encontraron supernovas desde {min_year_limit} en adelante")
            return
        
        if current_year < min_year:
            print(f"[INFO] Usando año mínimo: {current_year} (solicitado: {min_year})")
        
        # filter_by_year ya retorna filepaths directamente, no tuplas
        filtered_files = filtered_files_tuples
        
        # Seleccionar aleatoriamente TODOS los archivos disponibles (o al menos más de los necesarios)
        # Esto permite continuar intentando hasta tener n_supernovas exitosas
        import random
        seed_value = MCMC_CONFIG.get("random_seed", 42)
        random.seed(seed_value)  # Semilla fija para reproducibilidad
        # Mezclar todos los archivos aleatoriamente
        selected_files = filtered_files.copy()
        random.shuffle(selected_files)
    else:
        # Modo CSV: ya tenemos selected_files filtrados arriba
        # No necesitamos shuffle, mantener orden del CSV
        selected_files = filtered_files
    
    if supernovas_from_csv:
        print(f"\n[INFO] Procesando {len(selected_files)} supernovas del CSV...\n")
    else:
        print(f"\n[INFO] Encontrados {len(filtered_files)} archivos del año {current_year} en adelante")
        if n_supernovas is None:
            print(f"[INFO] Procesando TODAS las supernovas disponibles...\n")
        else:
            print(f"[INFO] Procesando hasta obtener {n_supernovas} supernovas exitosas...\n")
    
    # Determinar qué filtros procesar
    if filters_to_process is None:
        filters_to_process = FILTERS_TO_PROCESS
    
    if not filters_to_process:
        # Si está vacía, procesar todos los filtros disponibles (pero limitar a 2 para el PDF)
        if selected_files:
            test_filters, _ = parse_photometry_file(str(selected_files[0]))
            filters_to_process = sorted(list(test_filters.keys()))[:2]  # Máximo 2 filtros
            print(f"[INFO] Procesando filtros: {', '.join(filters_to_process)}")
    
    # Procesar supernovas y generar PDF
    # Continuar hasta tener n_supernovas exitosas
    processed_count = len(processed_supernovas)  # Empezar desde las ya procesadas
    failed_count = 0
    attempted_count = 0
    successful_supernovas = list(processed_supernovas)  # Empezar con las ya procesadas
    failed_supernovas = []  # Lista de tuplas: (sn_name, reason)
    
    # NO usar PdfPages con contexto - abriremos y cerraremos después de cada supernova
    # para evitar acumular páginas en memoria
    for filepath in selected_files:
            # Si ya tenemos las suficientes exitosas, parar (solo si n_supernovas no es None)
            if n_supernovas is not None and processed_count >= n_supernovas:
                break
            
            # Leer nombre de supernova para verificar checkpoint
            try:
                _, sn_name_test = parse_photometry_file(str(filepath))
            except:
                sn_name_test = Path(filepath).stem.replace('_photometry', '')
            
            # Saltar si ya está procesada
            if sn_name_test in processed_supernovas:
                if n_supernovas is None:
                    print(f"\n[SKIP] {Path(filepath).name} ya procesada (checkpoint) - Exitosas: {processed_count}")
                else:
                    print(f"\n[SKIP] {Path(filepath).name} ya procesada (checkpoint) - Exitosas: {processed_count}/{n_supernovas}")
                continue
            
            attempted_count += 1
            if n_supernovas is None:
                print(f"\n[Intento {attempted_count}/{len(selected_files)}] Procesando: {Path(filepath).name} (Exitosas: {processed_count})")
            else:
                print(f"\n[Intento {attempted_count}/{len(selected_files)}] Procesando: {Path(filepath).name} (Exitosas: {processed_count}/{n_supernovas})")
            
            try:
                # Leer archivo
                filters_data, sn_name = parse_photometry_file(str(filepath))
                
                if not filters_data:
                    print(f"  [SKIP] No se pudieron extraer datos")
                    failed_count += 1
                    failed_supernovas.append((sn_name_test, "No se pudieron extraer datos del archivo"))
                    del filters_data
                    gc.collect()  # Liberar memoria inmediatamente en caso de error
                    continue
                
                # Procesar cada filtro y generar figuras
                filter_figs = {}  # {filter_name: (fit_fig, corner_fig)}
                skip_reasons = []  # Para acumular razones de skip
                filter_data_dict = {}  # Guardar datos de cada filtro para calcular rango común
                
                for filter_name in filters_to_process:
                    if filter_name not in filters_data:
                        skip_reasons.append(f"Filtro {filter_name} no disponible en datos")
                        continue
                    
                    try:
                        # Preparar datos (igual que en process_single_filter)
                        lc_data = prepare_lightcurve(
                            filters_data[filter_name],
                            filter_name,
                            max_days_after_peak=DATA_FILTER_CONFIG["max_days_after_peak"],
                            max_days_before_peak=DATA_FILTER_CONFIG["max_days_before_peak"],
                            max_days_before_first_obs=DATA_FILTER_CONFIG["max_days_before_first_obs"]
                        )
                        
                        if lc_data is None:
                            # Contar detecciones antes de prepare_lightcurve para saber por qué falló
                            df_normal = filters_data[filter_name][~filters_data[filter_name]['Upperlimit']]
                            n_detections = len(df_normal)
                            skip_reasons.append(f"Filtro {filter_name}: {n_detections} detecciones (mínimo 7 requerido)")
                            continue
                        
                        phase = lc_data['phase']
                        mjd = lc_data['mjd']  # MJD original para plotear
                        flux = lc_data['flux']
                        flux_err = lc_data['flux_err']
                        mag = lc_data['mag']
                        mag_err = lc_data['mag_err']
                        is_upper_limit = lc_data.get('is_upper_limit', None)
                        had_upper_limits = lc_data.get('had_upper_limits', False)
                        filter_reference_mjd = lc_data.get('reference_mjd', None)
                        
                        # Verificar que haya al menos 7 detecciones (excluyendo upper limits)
                        # El modelo tiene 6 parámetros, necesitamos al menos 7 puntos para un ajuste determinado (n > p)
                        n_detections = len(phase) if is_upper_limit is None else np.sum(~is_upper_limit)
                        if n_detections < 7:
                            skip_reasons.append(f"Filtro {filter_name}: {n_detections} detecciones después de filtrado (mínimo 7)")
                            continue
                        
                        # Ajuste MCMC (usa fase relativa por filtro, que está bien)
                        mcmc_results = fit_mcmc(phase, flux, flux_err, verbose=False, is_upper_limit=is_upper_limit)
                        
                        # Validar comportamiento físico del fit
                        is_valid, reason = validate_physical_fit(mcmc_results, phase, flux, is_upper_limit)
                        if not is_valid:
                            skip_reasons.append(f"Filtro {filter_name}: Fit no físico - {reason}")
                            continue
                        
                        # Extraer features (igual que en modo normal)
                        features = extract_features(mcmc_results, phase, flux, flux_err, sn_name, filter_name)
                        features['sn_type'] = sn_type
                        
                        # Guardar features incrementalmente en archivo separado para debug
                        save_features_incremental([features], sn_type, debug_mode=True, from_csv=(supernovas_from_csv is not None), overwrite_pdf=overwrite_pdf)
                        
                        # Convertir flujo del modelo a magnitud
                        from model import flux_to_mag
                        model_flux_clipped = np.clip(mcmc_results['model_flux'], 1e-10, None)
                        mag_model = flux_to_mag(model_flux_clipped)
                        del model_flux_clipped  # Liberar inmediatamente
                        
                        # CONVERTIR MODELO A MJD PARA EL PLOT
                        # El modelo fue ajustado en fase relativa, necesitamos convertirlo a MJD
                        if filter_reference_mjd is not None:
                            # Ajustar samples: t0 está en fase relativa, convertirlo a MJD absoluto
                            from model import alerce_model
                            # Modificar in-place para evitar duplicar memoria (samples puede ser muy grande)
                            # Usar samples_valid para mediana/promedio, pero todos los samples para visualización
                            samples_to_use_debug = mcmc_results.get('samples_valid', mcmc_results['samples'])
                            samples_for_plot = samples_to_use_debug.copy()  # Necesitamos copia para no modificar original
                            samples_for_plot[:, 2] = samples_for_plot[:, 2] + filter_reference_mjd  # t0 en MJD absoluto
                            
                            # Usar EXACTAMENTE mcmc_results['params'] (la misma mediana que el corner plot)
                            # Solo convertir t0 a MJD para el eje X
                            param_medians_mjd = mcmc_results['params'].copy()
                            param_medians_mjd[2] = param_medians_mjd[2] + filter_reference_mjd  # t0 a MJD
                            flux_model_points_mjd = alerce_model(mjd, *param_medians_mjd)
                            flux_model_points_mjd = np.clip(flux_model_points_mjd, 1e-10, None)
                            mag_model_points_mjd = flux_to_mag(flux_model_points_mjd)
                            
                            # Convertir params_median_of_curves a MJD también
                            params_moc = mcmc_results.get('params_median_of_curves', None)
                            if params_moc is not None:
                                params_moc_mjd = params_moc.copy()
                                params_moc_mjd[2] = params_moc_mjd[2] + filter_reference_mjd  # t0 a MJD
                            else:
                                params_moc_mjd = None
                            
                            # Usar MJD y samples ajustados para el plot
                            phase_for_plot = mjd
                            mag_model_for_plot = mag_model_points_mjd
                            flux_model_for_plot = flux_model_points_mjd
                        else:
                            # Fallback: usar fase original
                            # Usar samples_valid si está disponible para mediana/promedio
                            samples_to_use_debug = mcmc_results.get('samples_valid', mcmc_results['samples'])
                            phase_for_plot = phase
                            mag_model_for_plot = mag_model
                            flux_model_for_plot = mcmc_results['model_flux']
                            samples_for_plot = samples_to_use_debug
                            param_medians_mjd = None  # No hay conversión a MJD
                            params_moc_mjd = mcmc_results.get('params_median_of_curves', None)
                        
                        # Guardar datos para calcular rango común si hay múltiples filtros
                        # IMPORTANTE: Guardar solo lo necesario, no todo mcmc_results
                        # Para corner plot, usar samples originales en fase relativa (no convertidos a MJD)
                        # Los parámetros deben estar en fase relativa para consistencia entre supernovas
                        filter_data_dict[filter_name] = {
                            'phase_for_plot': phase_for_plot,
                            'mag': mag,
                            'mag_err': mag_err,
                            'mag_model_for_plot': mag_model_for_plot,
                            'flux': flux,
                            'flux_model_for_plot': flux_model_for_plot,
                            'samples_for_plot': samples_for_plot,  # Samples convertidos a MJD para el plot
                            'is_upper_limit': is_upper_limit,
                            'flux_err': flux_err,
                            'had_upper_limits': had_upper_limits,
                            'mcmc_samples': mcmc_results.get('samples_valid', mcmc_results['samples']),  # Samples válidos para corner plot (mismos que param_medians)
                            'mcmc_results': mcmc_results,  # Guardar mcmc_results completo para plot_extended_model
                            'param_medians_mjd': param_medians_mjd if filter_reference_mjd is not None else None,  # Parámetros en MJD si aplica
                            'params_moc_mjd': params_moc_mjd,  # Params Median of Curves en MJD para curva verde
                            'filter_reference_mjd': filter_reference_mjd  # MJD de referencia para conversión
                        }
                        
                        # NO liberar mcmc_results todavía - se necesita para plot_extended_model
                        # Se liberará después de generar todos los gráficos
                        # Liberar lc_data y otros datos intermedios inmediatamente
                        del lc_data, phase, mjd, flux, flux_err, mag, mag_err
                        if filter_reference_mjd is not None:
                            del param_medians_mjd, flux_model_points_mjd, mag_model_points_mjd
                        del mag_model, filter_reference_mjd
                        
                    except Exception as e:
                        skip_reasons.append(f"Filtro {filter_name}: Error - {type(e).__name__}: {str(e)}")
                        import traceback
                        continue
                
                # Calcular rango común de MJD si hay 2 filtros
                common_xlim = None
                if len(filter_data_dict) == 2:
                    all_mjd_min = []
                    all_mjd_max = []
                    for filter_name, data in filter_data_dict.items():
                        mjd_data = data['phase_for_plot']
                        if len(mjd_data) > 0 and mjd_data.min() > 50000:  # Es MJD
                            all_mjd_min.append(mjd_data.min())
                            all_mjd_max.append(mjd_data.max())
                    if len(all_mjd_min) > 0:
                        common_xlim = (min(all_mjd_min), max(all_mjd_max))
                        # Agregar un pequeño margen (2% a cada lado)
                        mjd_range = common_xlim[1] - common_xlim[0]
                        common_xlim = (common_xlim[0] - 0.02 * mjd_range, common_xlim[1] + 0.02 * mjd_range)
                
                # Generar figuras con rango común si aplica
                for filter_name, data in filter_data_dict.items():
                    # plot_fit_with_uncertainty ahora retorna (fig, sample_indices, central_curve_idx)
                    # Usar param_medians_mjd si existe, sino mcmc_results['params']
                    # Esto asegura que la línea azul use EXACTAMENTE los mismos valores que el corner plot
                    if data.get('param_medians_mjd') is not None:
                        param_medians_for_plot = data['param_medians_mjd']
                    else:
                        param_medians_for_plot = data['mcmc_results']['params']
                    
                    fit_fig, sample_indices, central_curve_idx = plot_fit_with_uncertainty(
                        data['phase_for_plot'], data['mag'], data['mag_err'], 
                        data['mag_model_for_plot'], data['flux'], data['flux_model_for_plot'],
                        data['samples_for_plot'], n_samples_to_show=100,
                        sn_name=sn_name, filter_name=filter_name, save_path=None,
                        is_upper_limit=data['is_upper_limit'], 
                        flux_err=data['flux_err'],
                        had_upper_limits=data['had_upper_limits'],
                        xlim=common_xlim,
                        param_medians_phase_relative=data['mcmc_results']['params'],  # Para debug
                        param_medians=param_medians_for_plot,  # Mediana real para la línea azul
                        params_median_of_curves=data.get('params_moc_mjd'),  # Curva central (verde)
                        dynamic_bounds=data['mcmc_results'].get('dynamic_bounds', None)  # Bounds dinámicos del ajuste
                    )
                    
                    # Generar corner plot (sin guardar, solo obtener la figura)
                    # Pasar param_medians y param_percentiles para que muestre los valores de TODOS los samples
                    # Usar las 200 mejores curvas para el corner plot (consistente con param_medians)
                    samples_for_corner = data['mcmc_results'].get('samples_best_200', data.get('mcmc_samples'))
                    corner_fig = plot_corner(
                        samples_for_corner, 
                        save_path=None,
                        param_medians=data['mcmc_results']['params'],
                        param_percentiles=data['mcmc_results']['params_percentiles']
                    )
                    
                    # Generar modelo extendido (sin guardar, solo obtener la figura)
                    # Usamos el mismo sample de la curva central que en el plot normal
                    if data.get('param_medians_mjd') is not None:
                        params_for_extended = data['param_medians_mjd']
                    else:
                        params_for_extended = data['mcmc_results']['params']
                    
                    extended_fig = plot_extended_model(
                        data['phase_for_plot'], data['flux'], params_for_extended,
                        is_upper_limit=data['is_upper_limit'],
                        flux_err=data.get('flux_err', None),
                        sn_name=sn_name, filter_name=filter_name, save_path=None,
                        samples=data['samples_for_plot'],
                        precalculated_sample_indices=sample_indices,
                        central_curve_sample_idx=central_curve_idx  # Mismo sample que en plot normal
                    )
                    
                    filter_figs[filter_name] = (fit_fig, corner_fig, extended_fig)
                    
                    # NO eliminar datos todavía - se necesitan para prior/likelihood/posterior
                    # Se eliminarán después de generar todos los gráficos
                
                if not filter_figs:
                    print(f"  [SKIP] No se pudieron procesar filtros:")
                    for reason in skip_reasons:
                        print(f"    - {reason}")
                    failed_count += 1
                    # Combinar todas las razones de skip en una sola cadena
                    combined_reason = "; ".join(skip_reasons) if skip_reasons else "Ningún filtro pudo ser procesado"
                    failed_supernovas.append((sn_name, combined_reason))
                    # Liberar memoria antes de continuar
                    try:
                        del filter_figs, filter_data_dict
                        for filter_name in filters_to_process:
                            if filter_name in filters_data:
                                del filters_data[filter_name]
                        del filters_data
                    except:
                        pass
                    gc.collect()  # Forzar recolección inmediata
                    # No aumentar processed_count, seguir buscando
                    continue
                
                # Organizar figuras en una página del PDF
                n_filters = len(filter_figs)
                
                if n_filters == 1:
                    # Una sola página con: fit arriba, modelo extendido en medio, corner abajo
                    filter_name = list(filter_figs.keys())[0]
                    fit_fig, corner_fig, extended_fig = filter_figs[filter_name]
                    
                    # Crear figura combinada con 3 subplots verticales
                    if extended_fig is not None:
                        combined_fig = plt.figure(figsize=(10, 20))
                        gs = combined_fig.add_gridspec(3, 1, height_ratios=[1.3, 0.8, 1], hspace=0.05, 
                                                       left=0.08, right=0.95, top=0.96, bottom=0.05)
                    else:
                        combined_fig = plt.figure(figsize=(10, 16))
                        gs = combined_fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.05, 
                                                       left=0.08, right=0.95, top=0.96, bottom=0.05)
                    
                    # Copiar fit plot como imagen
                    ax1 = combined_fig.add_subplot(gs[0])
                    ax1.axis('off')
                    buf1 = io.BytesIO()
                    fit_fig.savefig(buf1, format='png', dpi=200, bbox_inches='tight')
                    buf1.seek(0)
                    img1 = mpimg.imread(buf1)
                    ax1.imshow(img1, aspect='auto', interpolation='bilinear')
                    
                    # Copiar modelo extendido como imagen (si existe)
                    if extended_fig is not None:
                        ax2 = combined_fig.add_subplot(gs[1])
                        ax2.axis('off')
                        buf2 = io.BytesIO()
                        extended_fig.savefig(buf2, format='png', dpi=200, bbox_inches='tight')
                        buf2.seek(0)
                        img2 = mpimg.imread(buf2)
                        ax2.imshow(img2, aspect='auto', interpolation='bilinear')
                        corner_idx = 2
                    else:
                        corner_idx = 1
                    
                    # Copiar corner plot como imagen
                    ax3 = combined_fig.add_subplot(gs[corner_idx])
                    ax3.axis('off')
                    buf3 = io.BytesIO()
                    corner_fig.savefig(buf3, format='png', dpi=200, bbox_inches='tight')
                    buf3.seek(0)
                    img3 = mpimg.imread(buf3)
                    ax3.imshow(img3, aspect='auto', interpolation='bilinear')
                    
                    combined_fig.suptitle(f'{sn_name} ({sn_type})', fontsize=14, fontweight='bold', y=0.98)
                    
                    # Guardar página al PDF
                    print(f"  [DEBUG] Guardando página al PDF para {sn_name} (1 filtro)")
                    _save_page_to_pdf(combined_fig, pdf_path, pdf_exists=pdf_exists)
                    pdf_exists = True
                    print(f"  [DEBUG] PDF existe después de guardar: {pdf_path.exists()}")
                    
                    # Liberar memoria
                    plt.close(combined_fig)
                    plt.close(fit_fig)
                    if extended_fig is not None:
                        plt.close(extended_fig)
                    plt.close(corner_fig)
                    del img1, img3
                    buf1.close()
                    buf3.close()
                    del buf1, buf3
                    if extended_fig is not None:
                        del img2
                        buf2.close()
                        del buf2
                    plt.close('all')
                    
                elif n_filters == 2:
                    # Una sola página con: fit1, fit2, extended1, extended2, corner1, corner2
                    filter_names = sorted(filter_figs.keys())
                    fit_fig1, corner_fig1, extended_fig1 = filter_figs[filter_names[0]]
                    fit_fig2, corner_fig2, extended_fig2 = filter_figs[filter_names[1]]
                    
                    # Determinar número de subplots según si hay modelos extendidos
                    n_subplots = 4  # fit1, fit2, corner1, corner2
                    if extended_fig1 is not None:
                        n_subplots += 1
                    if extended_fig2 is not None:
                        n_subplots += 1
                    
                    # Crear figura combinada
                    combined_fig = plt.figure(figsize=(10, 6 * n_subplots))
                    
                    # Definir height_ratios
                    height_ratios = [1.3, 1.3]  # fit1, fit2
                    if extended_fig1 is not None:
                        height_ratios.append(0.8)  # extended1
                    if extended_fig2 is not None:
                        height_ratios.append(0.8)  # extended2
                    height_ratios.extend([1, 1])  # corner1, corner2
                    
                    gs = combined_fig.add_gridspec(n_subplots, 1, height_ratios=height_ratios, hspace=0.05,
                                                   left=0.08, right=0.95, top=0.98, bottom=0.04)
                    
                    idx = 0
                    
                    # Fit 1
                    ax = combined_fig.add_subplot(gs[idx])
                    ax.axis('off')
                    buf = io.BytesIO()
                    fit_fig1.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                    buf.seek(0)
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                    buf.close()
                    plt.close(fit_fig1)  # Liberar inmediatamente
                    idx += 1
                    
                    # Fit 2
                    ax = combined_fig.add_subplot(gs[idx])
                    ax.axis('off')
                    buf = io.BytesIO()
                    fit_fig2.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                    buf.seek(0)
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                    buf.close()
                    plt.close(fit_fig2)  # Liberar inmediatamente
                    idx += 1
                    
                    # Extended 1 (si existe)
                    if extended_fig1 is not None:
                        ax = combined_fig.add_subplot(gs[idx])
                        ax.axis('off')
                        buf = io.BytesIO()
                        extended_fig1.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                        buf.seek(0)
                        img = mpimg.imread(buf)
                        ax.imshow(img, aspect='auto', interpolation='bilinear')
                        buf.close()
                        plt.close(extended_fig1)  # Liberar inmediatamente
                        idx += 1
                    
                    # Extended 2 (si existe)
                    if extended_fig2 is not None:
                        ax = combined_fig.add_subplot(gs[idx])
                        ax.axis('off')
                        buf = io.BytesIO()
                        extended_fig2.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                        buf.seek(0)
                        img = mpimg.imread(buf)
                        ax.imshow(img, aspect='auto', interpolation='bilinear')
                        buf.close()
                        plt.close(extended_fig2)  # Liberar inmediatamente
                        idx += 1
                    
                    # Corner 1 (DPI bajo para ahorrar memoria)
                    ax = combined_fig.add_subplot(gs[idx])
                    ax.axis('off')
                    buf = io.BytesIO()
                    corner_fig1.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                    buf.seek(0)
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                    buf.close()
                    plt.close(corner_fig1)  # Liberar inmediatamente
                    del corner_fig1  # Liberar referencia
                    idx += 1
                    
                    # Corner 2 (DPI bajo para ahorrar memoria)
                    ax = combined_fig.add_subplot(gs[idx])
                    ax.axis('off')
                    buf = io.BytesIO()
                    corner_fig2.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                    buf.seek(0)
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                    buf.close()
                    plt.close(corner_fig2)  # Liberar inmediatamente
                    del corner_fig2  # Liberar referencia
                    
                    # Forzar liberación de memoria antes de guardar
                    gc.collect()
                    
                    combined_fig.suptitle(f'{sn_name} ({sn_type})', fontsize=14, fontweight='bold', y=0.99)
                    
                    # Guardar página al PDF
                    print(f"  [DEBUG] Guardando página al PDF para {sn_name} (2 filtros)")
                    _save_page_to_pdf(combined_fig, pdf_path, pdf_exists=pdf_exists)
                    pdf_exists = True
                    print(f"  [DEBUG] PDF existe después de guardar: {pdf_path.exists()}")
                    
                    # Liberar memoria
                    plt.close(combined_fig)
                    plt.close('all')
                    gc.collect()
                
                # Liberar memoria: eliminar datos grandes inmediatamente
                # Primero liberar datos individuales de filter_data_dict
                for filter_name in filter_data_dict:
                    data = filter_data_dict[filter_name]
                    if 'mcmc_samples' in data:
                        del data['mcmc_samples']
                    if 'samples_for_plot' in data:
                        del data['samples_for_plot']
                    if 'mcmc_results' in data:
                        # Liberar samples dentro de mcmc_results si existe
                        if isinstance(data['mcmc_results'], dict) and 'samples' in data['mcmc_results']:
                            del data['mcmc_results']['samples']
                        del data['mcmc_results']
                    if 'phase_for_plot' in data:
                        del data['phase_for_plot']
                    if 'mag' in data:
                        del data['mag'], data['mag_err']
                    if 'mag_model_for_plot' in data:
                        del data['mag_model_for_plot'], data['flux_model_for_plot']
                    if 'flux_err' in data:
                        del data['flux_err']
                del filter_figs, filter_data_dict
                # Limpiar referencias a datos de filtros
                for filter_name in filters_to_process:
                    if filter_name in filters_data:
                        del filters_data[filter_name]
                del filters_data
                
                # Forzar recolección de basura después de cada supernova
                gc.collect()
                
                processed_count += 1
                successful_supernovas.append(sn_name)  # Guardar nombre de supernova exitosa
                processed_supernovas.add(sn_name)  # Añadir al set de procesadas
                
                # Guardar checkpoint después de cada supernova exitosa
                save_debug_checkpoint(sn_type, processed_supernovas)
                
                if n_supernovas is None:
                    print(f"  [OK] Página agregada al PDF y checkpoint guardado ({processed_count} exitosas)")
                else:
                    print(f"  [OK] Página agregada al PDF y checkpoint guardado ({processed_count}/{n_supernovas} exitosas)")
                
                # Liberar memoria periódicamente cada 5 supernovas exitosas (más frecuente)
                if processed_count % 5 == 0:
                    gc.collect()
                    print(f"  [INFO] Memoria liberada después de procesar {processed_count} supernovas exitosas")
                
            except Exception as e:
                print(f"  [ERROR] Error procesando supernova: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                # Capturar la razón del error
                error_reason = f"Error: {type(e).__name__}: {str(e)}"
                failed_supernovas.append((sn_name if 'sn_name' in locals() else sn_name_test, error_reason))
                # Liberar memoria en caso de error
                try:
                    del filter_figs, filter_data_dict
                    if 'filters_data' in locals():
                        for filter_name in filters_to_process:
                            if filter_name in filters_data:
                                del filters_data[filter_name]
                        del filters_data
                except:
                    pass
                plt.close('all')  # Cerrar todas las figuras de matplotlib
                gc.collect()  # Forzar recolección inmediata
                continue
    
    # No necesitamos combinar PDFs al final - ya se guardaron incrementalmente
    # Guardar CSV con supernovas exitosas
    if supernovas_from_csv and csv_file_path:
        # Si se usa CSV, siempre usar sufijo _from_csv (independiente de --overwrite)
        csv_filename = sn_type.replace(' ', '_').replace('-', '_') + '_successful_from_csv.csv'
    else:
        csv_filename = sn_type.replace(' ', '_').replace('-', '_') + '_successful.csv'
    csv_path = DEBUG_PDF_DIR / csv_filename
    
    # Si se usa --overwrite, borrar el CSV existente antes de escribir
    if overwrite_pdf and csv_path.exists():
        try:
            csv_path.unlink()
            print(f"[INFO] CSV existente eliminado (modo overwrite)")
        except Exception as e:
            print(f"[WARNING] No se pudo eliminar el CSV existente: {e}")
    
    df_successful = pd.DataFrame({'supernova_name': successful_supernovas})
    df_successful.to_csv(csv_path, index=False)
    
    # Guardar CSV con supernovas fallidas y sus razones
    if supernovas_from_csv and csv_file_path:
        failed_csv_filename = sn_type.replace(' ', '_').replace('-', '_') + '_failed_from_csv.csv'
    else:
        failed_csv_filename = sn_type.replace(' ', '_').replace('-', '_') + '_failed.csv'
    failed_csv_path = DEBUG_PDF_DIR / failed_csv_filename
    
    # Si se usa --overwrite, borrar el CSV de fallidas existente
    if overwrite_pdf and failed_csv_path.exists():
        try:
            failed_csv_path.unlink()
        except Exception as e:
            print(f"[WARNING] No se pudo eliminar el CSV de fallidas existente: {e}")
    
    if failed_supernovas:
        df_failed = pd.DataFrame(failed_supernovas, columns=['supernova_name', 'reason'])
        df_failed.to_csv(failed_csv_path, index=False)
    
    # Calcular estadísticas por tipo de fallo
    fail_stats = {}
    for sn_name, reason in failed_supernovas:
        # Categorizar la razón
        if "detecciones" in reason.lower() or "mínimo" in reason.lower():
            category = "Pocos datos (<7 detecciones)"
        elif "fit no físico" in reason.lower():
            category = "Fit no físico"
        elif "no disponible" in reason.lower():
            category = "Filtro no disponible"
        elif "extraer datos" in reason.lower():
            category = "Error leyendo archivo"
        elif "error:" in reason.lower():
            category = "Error de procesamiento"
        else:
            category = "Otros"
        fail_stats[category] = fail_stats.get(category, 0) + 1
    
    # Mostrar resumen
    print(f"\n{'='*80}")
    print(f"RESUMEN DE PROCESAMIENTO")
    print(f"{'='*80}")
    total_attempted = processed_count + failed_count
    success_rate = (processed_count / total_attempted * 100) if total_attempted > 0 else 0
    fail_rate = (failed_count / total_attempted * 100) if total_attempted > 0 else 0
    
    print(f"Total intentadas: {total_attempted}")
    print(f"Exitosas: {processed_count} ({success_rate:.1f}%)")
    print(f"Fallidas: {failed_count} ({fail_rate:.1f}%)")
    
    if fail_stats:
        print(f"\nRazones de fallo:")
        for category, count in sorted(fail_stats.items(), key=lambda x: -x[1]):
            pct = (count / failed_count * 100) if failed_count > 0 else 0
            print(f"  - {category}: {count} ({pct:.1f}%)")
    
    print(f"\nArchivos generados:")
    print(f"  - PDF: {pdf_path}")
    print(f"  - Exitosas: {csv_path}")
    if failed_supernovas:
        print(f"  - Fallidas: {failed_csv_path}")
    
    if n_supernovas is not None and processed_count < n_supernovas:
        print(f"\n[WARNING] Solo se procesaron {processed_count} de {n_supernovas} supernovas solicitadas")
    print(f"{'='*80}")


def main():
    """
    Función principal
    
    Uso:
        python main.py "SN Ia" 3 ["g", "r"]
        
    Argumentos:
        1. Tipo de supernova (ej: "SN Ia")
        2. Número de supernovas a procesar (ej: 3) o "all" para todas
        3. Lista de filtros (opcional, ej: "g,r" o "g r")
           Si no se especifica, usa FILTERS_TO_PROCESS de config.py
        4. --resume : Continuar desde checkpoint (opcional)
        5. --seed <número> o --random-seed <número> : Semilla aleatoria para reproducibilidad (opcional)
           Si no se especifica, usa el valor de MCMC_CONFIG["random_seed"] en config.py (por defecto: 42)
    """
    # Parsear argumentos
    if len(sys.argv) > 1:
        sn_type = sys.argv[1]
    else:
        sn_type = "SN Ia"
    
    # Verificar si hay flag --resume o --debug-pdf
    # Buscar en todos los argumentos, incluso si están concatenados sin espacio
    resume_from_checkpoint = any('--resume' in arg for arg in sys.argv)
    debug_pdf_mode = any('--debug-pdf' in arg for arg in sys.argv)
    
    # Si está en modo debug-pdf, usar función especial
    if debug_pdf_mode:
        # Parsear número de supernovas para PDF (se ignorará si se usa --from-csv)
        n_supernovas = None
        if len(sys.argv) > 2 and sys.argv[2] != '--debug-pdf':
            if sys.argv[2].lower() == 'all':
                n_supernovas = None
            else:
                n_supernovas = int(sys.argv[2])
        else:
            # Valores por defecto según tipo
            if sn_type == "SN Ia":
                n_supernovas = 200
            elif sn_type in ["SN Ia-91bg-like", "SN Ia-91T-like"]:
                n_supernovas = 50
            else:
                n_supernovas = 100
        
        # Parsear --from-csv o --csv-file (opcional)
        csv_file_path = None
        overwrite_pdf = False
        for i, arg in enumerate(sys.argv):
            if arg == '--from-csv' or arg == '--csv-file':
                if i + 1 < len(sys.argv):
                    csv_file_path = sys.argv[i + 1]
                    print(f"[INFO] Modo CSV activado: procesando supernovas del archivo: {csv_file_path}")
                    # Si se especifica CSV, ignorar n_supernovas
                    n_supernovas = None
                else:
                    print(f"[ERROR] --from-csv requiere un archivo CSV como argumento")
                    return
                break
        
        # Parsear --overwrite (opcional, solo tiene efecto con --from-csv)
        if any('--overwrite' in arg for arg in sys.argv):
            overwrite_pdf = True
            print(f"[INFO] Modo overwrite activado: se sobrescribirá el PDF si existe")
        
        # Parsear filtros (opcional)
        filters_to_process = None
        for arg in sys.argv[3:]:
            # Ignorar flags y argumentos que contengan flags
            if (arg == '--debug-pdf' or arg == '--resume' or 
                '--debug-pdf' in arg or '--resume' in arg or
                arg.startswith('--seed') or arg.startswith('--random-seed') or
                arg == '--from-csv' or arg == '--csv-file' or
                arg == '--overwrite' or '--overwrite' in arg or
                (csv_file_path and arg == csv_file_path)):
                continue
            filters_input = arg
            filters_input = filters_input.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
            filters_to_process = [f.strip() for f in filters_input.replace(',', ' ').split()]
            print(f"[INFO] Filtros especificados: {filters_to_process}")
            break
        
        # Parsear semilla aleatoria (opcional) para modo debug-pdf
        for i, arg in enumerate(sys.argv):
            if arg == '--seed' or arg == '--random-seed':
                if i + 1 < len(sys.argv) and sys.argv[i + 1] != csv_file_path:
                    try:
                        random_seed = int(sys.argv[i + 1])
                        MCMC_CONFIG["random_seed"] = random_seed
                        print(f"[INFO] Semilla aleatoria especificada: {random_seed}")
                    except (ValueError, IndexError):
                        print(f"[WARNING] Valor inválido para --seed, usando valor por defecto: {MCMC_CONFIG.get('random_seed', 'No configurada')}")
                break
        
        # Cargar lista de supernovas del CSV si se especificó
        supernovas_from_csv = None
        if csv_file_path:
            try:
                supernovas_from_csv = load_supernovas_from_csv(csv_file_path)
            except Exception as e:
                print(f"[ERROR] No se pudo leer el CSV: {e}")
                return
        
        generate_debug_pdf(sn_type, n_supernovas, filters_to_process, min_year=2022, 
                         resume_from_checkpoint=resume_from_checkpoint, 
                         supernovas_from_csv=supernovas_from_csv,
                         csv_file_path=csv_file_path,
                         overwrite_pdf=overwrite_pdf)
        return
    
    # Parsear número de supernovas
    n_supernovas = None
    if len(sys.argv) > 2 and sys.argv[2] != '--resume':
        if sys.argv[2].lower() == 'all':
            n_supernovas = None  # Procesar todas
        else:
            n_supernovas = int(sys.argv[2])
    else:
        n_supernovas = 3
    
    # Parsear filtros (opcional)
    filters_to_process = None
    for arg in sys.argv[3:]:
        if arg == '--resume' or arg.startswith('--seed') or arg.startswith('--random-seed'):
            continue
        filters_input = arg
        # Aceptar formato "g,r" o "g r" o "['g','r']"
        filters_input = filters_input.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
        filters_to_process = [f.strip() for f in filters_input.replace(',', ' ').split()]
        print(f"[INFO] Filtros especificados: {filters_to_process}")
        break
    
    # Parsear semilla aleatoria (opcional)
    random_seed = None
    for i, arg in enumerate(sys.argv):
        if arg == '--seed' or arg == '--random-seed':
            if i + 1 < len(sys.argv):
                try:
                    random_seed = int(sys.argv[i + 1])
                    MCMC_CONFIG["random_seed"] = random_seed
                    print(f"[INFO] Semilla aleatoria especificada: {random_seed}")
                except (ValueError, IndexError):
                    print(f"[WARNING] Valor inválido para --seed, usando valor por defecto: {MCMC_CONFIG.get('random_seed', 'No configurada')}")
            break
    
    # Configurar logger
    logger = setup_logger(sn_type)
    
    # Cargar checkpoint si se solicita
    processed_set = None
    if resume_from_checkpoint:
        processed_set = load_checkpoint(sn_type)
        n_processed = len(processed_set)
        if n_processed > 0:
            print(f"[INFO] Checkpoint cargado: {n_processed} combinaciones ya procesadas")
            logger.info(f"Checkpoint cargado: {n_processed} combinaciones ya procesadas")
        else:
            print(f"[INFO] No se encontró checkpoint, comenzando desde el inicio")
            logger.info("No se encontró checkpoint, comenzando desde el inicio")
    
    print(f"\n{'='*80}")
    print(f"EXTRACCION DE FEATURES CON MCMC")
    print(f"{'='*80}")
    logger.info(f"Configuración: Tipo={sn_type}, N_supernovas={n_supernovas or 'TODAS'}, Filtros={filters_to_process or FILTERS_TO_PROCESS}")
    print(f"Tipo de supernova: {sn_type}")
    if n_supernovas is None:
        print(f"Numero de supernovas: TODAS")
    else:
        print(f"Numero de supernovas: {n_supernovas}")
    if filters_to_process:
        print(f"Filtros a procesar: {', '.join(filters_to_process)}")
    else:
        print(f"Filtros a procesar: {FILTERS_TO_PROCESS} (desde config.py)")
    print(f"Semilla aleatoria (reproducibilidad): {MCMC_CONFIG.get('random_seed', 'No configurada')}")
    if resume_from_checkpoint:
        print(f"Modo: RESUMIR desde checkpoint")
    else:
        print(f"Modo: NUEVO (usar --resume para continuar desde checkpoint)")
    print(f"Directorio de datos: {BASE_DATA_PATH}")
    print(f"Directorio de salida: {PLOTS_DIR.parent}")
    
    # Buscar archivos
    type_path = BASE_DATA_PATH / sn_type
    
    if not type_path.exists():
        print(f"\n[ERROR] La carpeta '{sn_type}' no existe")
        return
    
    dat_files = list(type_path.glob("*_photometry.dat"))
    
    if not dat_files:
        print(f"\n[ERROR] No se encontraron archivos .dat en '{sn_type}'")
        return
    
    print(f"\n[INFO] Encontrados {len(dat_files)} archivos")
    if n_supernovas is None:
        print(f"[INFO] Procesando TODAS las supernovas...\n")
    else:
        print(f"[INFO] Procesando primeros {min(n_supernovas, len(dat_files))} archivos...\n")
    
    # Inicializar checkpoint si no existe
    if processed_set is None:
        processed_set = load_checkpoint(sn_type)
        if len(processed_set) > 0:
            print(f"[INFO] Checkpoint existente cargado: {len(processed_set)} combinaciones ya procesadas")
    
    # Procesar supernovas
    all_features = []
    total_processed = 0
    t0_total = time.time()
    
    # Determinar cuántas procesar
    files_to_process = dat_files if n_supernovas is None else dat_files[:n_supernovas]
    
    logger.info(f"Iniciando procesamiento de {len(files_to_process)} supernovas")
    
    for i, filepath in enumerate(files_to_process):
        logger.info(f"Procesando supernova {i+1}/{len(files_to_process)}: {Path(filepath).name}")
        features_list = process_supernova(str(filepath), sn_type, filters_to_process, processed_set, logger)
        all_features.extend(features_list)  # Extender con todas las features de todos los filtros
        total_processed += len(features_list)
        
        # Guardar features incrementalmente después de procesar cada supernova
        if features_list:
            save_features_incremental(features_list, sn_type)
            print(f"  [OK] Features guardadas incrementalmente ({len(features_list)} registros)")
        
        # Liberar memoria periódicamente cada 10 supernovas
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"  [INFO] Memoria liberada después de procesar {i+1} supernovas")
    
    t_total = time.time() - t0_total
    
    # Determinar si todo estaba en checkpoint o hubo un error real
    # Si no hay features nuevas pero hay checkpoint, probablemente todo estaba procesado
    if len(all_features) == 0 and processed_set and len(processed_set) > 0:
        # Todo estaba en checkpoint, no es un error
        logger.info(f"Procesamiento completado. Todas las supernovas ya estaban procesadas en checkpoint")
        logger.info(f"Tiempo total: {t_total:.2f}s")
    else:
        logger.info(f"Procesamiento completado. Total: {len(all_features)} features extraídas ({total_processed} procesados) en {t_total:.2f}s")
    
    # Mostrar resumen de features guardadas (ya se guardaron incrementalmente)
    if all_features:
        output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}.csv"
        
        # Leer el archivo final para mostrar estadísticas
        if output_file.exists():
            final_df = pd.read_csv(output_file)
            print(f"\n{'='*80}")
            print(f"[OK] Features guardadas incrementalmente en: {output_file}")
            print(f"[OK] Registros procesados en esta sesión: {len(all_features)}")
            print(f"[OK] Total de registros en archivo: {len(final_df)}")
            print(f"[OK] Supernovas únicas: {final_df['sn_name'].nunique()}")
            print(f"[OK] Filtros procesados: {', '.join(sorted(final_df['filter_band'].unique()))}")
        else:
            # Esto no debería pasar si se guardó incrementalmente, pero por si acaso
            print(f"\n{'='*80}")
            print(f"[WARNING] Archivo de features no encontrado. Esto no debería pasar.")
        
        print(f"[OK] Tiempo total de ejecución: {t_total:.2f} segundos ({t_total/60:.2f} minutos)")
        print(f"{'='*80}")
        logger.info(f"Ejecucion completada exitosamente. Features guardadas en: {output_file}")
    elif processed_set and len(processed_set) > 0:
        # Todo estaba en checkpoint, no es un error
        print(f"\n{'='*80}")
        print(f"[INFO] Todas las supernovas ya estaban procesadas en el checkpoint")
        print(f"[INFO] Combinaciones en checkpoint: {len(processed_set)}")
        print(f"[INFO] Tiempo total: {t_total:.2f} segundos ({t_total/60:.2f} minutos)")
        print(f"{'='*80}")
        logger.info(f"Ejecucion completada: Todas las supernovas ya estaban procesadas en checkpoint ({len(processed_set)} combinaciones)")
    else:
        # Error real: no se procesó nada y no fue por checkpoint
        error_msg = "No se pudieron procesar supernovas"
        print(f"\n[ERROR] {error_msg}")
        logger.error(f"ERROR: {error_msg}")

if __name__ == "__main__":
    main()

