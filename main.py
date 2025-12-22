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
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from reader import parse_photometry_file, prepare_lightcurve
from mcmc_fitter import fit_mcmc
from feature_extractor import extract_features
from plotter import plot_fit, plot_corner
from config import BASE_DATA_PATH, PLOTS_DIR, FEATURES_DIR, CHECKPOINT_DIR, LOG_DIR, FILTERS_TO_PROCESS, MCMC_CONFIG, DATA_FILTER_CONFIG

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
            max_days_before_peak=DATA_FILTER_CONFIG["max_days_before_peak"]
        )
        
        if lc_data is None:
            error_msg = f"No hay suficientes datos para filtro {filter_name}"
            print(f"    [ERROR] {error_msg}")
            if logger:
                logger.error(f"  [{sn_name} | {filter_name}] ERROR en Paso 1/5: {error_msg}")
            return None
        
        phase = lc_data['phase']
        flux = lc_data['flux']
        flux_err = lc_data['flux_err']
        mag = lc_data['mag']
        mag_err = lc_data['mag_err']
        peak_phase = lc_data.get('peak_phase', None)
        
        print(f"    [OK] Puntos de datos: {len(phase)}")
        print(f"    [OK] Rango de fase: {phase.min():.1f} - {phase.max():.1f} días")
        if peak_phase is not None:
            print(f"    [OK] Peak phase: {peak_phase:.1f} días")
            print(f"    [OK] Datos filtrados: {DATA_FILTER_CONFIG['max_days_before_peak']:.0f} días antes y {DATA_FILTER_CONFIG['max_days_after_peak']:.0f} días después del peak")
        
        # Ajuste MCMC
        if logger:
            logger.info(f"  [{sn_name} | {filter_name}] Paso 2/5: Ejecutando MCMC (walkers={MCMC_CONFIG['n_walkers']}, steps={MCMC_CONFIG['n_steps']})")
        print(f"    Ejecutando MCMC...")
        t0_mcmc = time.time()
        
        try:
            mcmc_results = fit_mcmc(phase, flux, flux_err, verbose=False)
            t_mcmc = time.time() - t0_mcmc
            print(f"    [OK] MCMC completado en {t_mcmc:.2f} segundos")
            if logger:
                logger.info(f"  [{sn_name} | {filter_name}] Paso 2/5: MCMC completado exitosamente en {t_mcmc:.2f}s")
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
            features = extract_features(mcmc_results, phase, mag, mag_err, mag_model,
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
        
        # Crear subcarpeta para esta supernova
        sn_plots_dir = PLOTS_DIR / sn_name
        sn_plots_dir.mkdir(exist_ok=True)
        
        # Guardar gráficos (con múltiples realizaciones MCMC)
        plot_filename = f"{sn_name}_{filter_name}_fit.png"
        plot_path = sn_plots_dir / plot_filename
        from plotter import plot_fit_with_uncertainty
        t0_plot = time.time()
        # Calcular número total de samples usados para la mediana
        n_total_samples = len(mcmc_results['samples'])
        print(f"    [INFO] Samples totales para mediana: {n_total_samples:,} (de {MCMC_CONFIG['n_walkers']} walkers × {MCMC_CONFIG['n_steps'] - MCMC_CONFIG['burn_in']} pasos)")
        plot_fit_with_uncertainty(
            phase, mag, mag_err, mag_model, flux, mcmc_results['model_flux'],
            mcmc_results['samples'], n_samples_to_show=100,  # Valor por defecto: 100 realizaciones para visualización
            sn_name=sn_name, filter_name=filter_name, save_path=str(plot_path)
        )
        t_plot = time.time() - t0_plot
        print(f"    [OK] Gráfico guardado en {t_plot:.2f} segundos: {plot_path}")
        
        # Corner plot
        corner_filename = f"{sn_name}_{filter_name}_corner.png"
        corner_path = sn_plots_dir / corner_filename
        t0_corner = time.time()
        plot_corner(mcmc_results['samples'], save_path=str(corner_path))
        t_corner = time.time() - t0_corner
        print(f"    [OK] Corner plot guardado en {t_corner:.2f} segundos: {corner_path}")
        
        # Liberar memoria: eliminar samples grandes del MCMC después de usarlos
        del mcmc_results['samples']
        import gc
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
    """
    # Parsear argumentos
    if len(sys.argv) > 1:
        sn_type = sys.argv[1]
    else:
        sn_type = "SN Ia"
    
    # Verificar si hay flag --resume
    resume_from_checkpoint = '--resume' in sys.argv
    
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
        if arg == '--resume':
            continue
        filters_input = arg
        # Aceptar formato "g,r" o "g r" o "['g','r']"
        filters_input = filters_input.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
        filters_to_process = [f.strip() for f in filters_input.replace(',', ' ').split()]
        print(f"[INFO] Filtros especificados: {filters_to_process}")
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
        
        # Liberar memoria periódicamente cada 10 supernovas
        if (i + 1) % 10 == 0:
            import gc
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
    
    # Guardar features en CSV
    if all_features:
        df_new_features = pd.DataFrame(all_features)
        output_file = FEATURES_DIR / f"features_{sn_type.replace(' ', '_')}.csv"
        
        # Si el archivo ya existe, leerlo y reemplazar solo las entradas específicas
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            
            # Identificar qué entradas se van a reemplazar (mismo sn_name y filter_band)
            for _, row in df_new_features.iterrows():
                mask = (existing_df['sn_name'] == row['sn_name']) & (existing_df['filter_band'] == row['filter_band'])
                if mask.any():
                    # Eliminar entradas existentes para esta combinación
                    existing_df = existing_df[~mask]
                    print(f"  [INFO] Reemplazando entrada existente: {row['sn_name']} - {row['filter_band']}")
            
            # Combinar: primero las existentes (sin duplicados), luego las nuevas
            combined_df = pd.concat([existing_df, df_new_features], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            
            print(f"\n{'='*80}")
            print(f"[OK] Features guardadas en: {output_file}")
            print(f"[OK] Nuevos registros agregados: {len(df_new_features)}")
            print(f"[OK] Total de registros en archivo: {len(combined_df)}")
            print(f"[OK] Supernovas únicas: {combined_df['sn_name'].nunique()}")
            print(f"[OK] Filtros procesados: {', '.join(combined_df['filter_band'].unique())}")
        else:
            # Si no existe, crear nuevo archivo
            df_new_features.to_csv(output_file, index=False)
            print(f"\n{'='*80}")
            print(f"[OK] Features guardadas en: {output_file}")
            print(f"[OK] Total de registros (supernovas x filtros): {len(all_features)}")
            print(f"[OK] Supernovas únicas: {df_new_features['sn_name'].nunique()}")
            print(f"[OK] Filtros procesados: {', '.join(df_new_features['filter_band'].unique())}")
        
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

