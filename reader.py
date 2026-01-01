"""
Lector de archivos .dat de fotometría ZTF
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

def parse_photometry_file(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parsear archivo .dat y extraer datos por filtro
    
    Returns:
    --------
    dict: {filtro: DataFrame con columnas [MJD, MAG, MAGERR, Upperlimit]}
    sn_name: str
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer nombre de supernova
    sn_name = None
    for line in lines[:10]:
        if 'SNNAME:' in line:
            sn_name = line.split('SNNAME:')[1].strip()
            break
    
    # Parsear secciones por filtro
    filters_data = {}
    current_filter = None
    current_data = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('# FILTER'):
            # Guardar filtro anterior
            if current_filter and current_data:
                df = pd.DataFrame(current_data, columns=['MJD', 'MAG', 'MAGERR', 'Upperlimit',
                                                          'Instrument', 'Telescope', 'Source'])
                df['MJD'] = pd.to_numeric(df['MJD'], errors='coerce')
                df['MAG'] = pd.to_numeric(df['MAG'], errors='coerce')
                df['MAGERR'] = pd.to_numeric(df['MAGERR'], errors='coerce')
                df['Upperlimit'] = df['Upperlimit'].str.strip().str.upper() == 'T'
                df = df[['MJD', 'MAG', 'MAGERR', 'Upperlimit']].dropna(subset=['MJD', 'MAG'])
                if len(df) > 0:
                    filters_data[current_filter] = df
            
            current_filter = line.split('FILTER')[1].strip()
            current_data = []
        
        elif current_filter and (line.startswith('\t') or (line and not line.startswith('#'))):
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    mjd = float(parts[0].strip())
                    mag = float(parts[1].strip())
                    magerr = float(parts[2].strip())
                    upperlimit = parts[3].strip() if len(parts) > 3 else 'F'
                    current_data.append([mjd, mag, magerr, upperlimit, 'nan', 'nan', 'nan'])
                except (ValueError, IndexError):
                    continue
    
    # Guardar último filtro
    if current_filter and current_data:
        df = pd.DataFrame(current_data, columns=['MJD', 'MAG', 'MAGERR', 'Upperlimit',
                                                  'Instrument', 'Telescope', 'Source'])
        df['MJD'] = pd.to_numeric(df['MJD'], errors='coerce')
        df['MAG'] = pd.to_numeric(df['MAG'], errors='coerce')
        df['MAGERR'] = pd.to_numeric(df['MAGERR'], errors='coerce')
        df['Upperlimit'] = df['Upperlimit'].str.strip().str.upper() == 'T'
        df = df[['MJD', 'MAG', 'MAGERR', 'Upperlimit']].dropna(subset=['MJD', 'MAG'])
        if len(df) > 0:
            filters_data[current_filter] = df
    
    return filters_data, sn_name

def apply_time_window_filter(df_normal: pd.DataFrame, window_hours: float = 8.0) -> pd.DataFrame:
    """
    Aplicar filtro de ventana temporal: agrupar puntos dentro de una ventana de tiempo
    y reemplazarlos por la mediana de la ventana.
    
    Parameters:
    -----------
    df_normal : pd.DataFrame
        DataFrame con columnas MJD, MAG, MAGERR (solo detecciones normales, sin upper limits)
    window_hours : float
        Tamaño de la ventana en horas (default: 8 horas)
    
    Returns:
    --------
    pd.DataFrame con columnas MJD, MAG, MAGERR (datos agrupados por ventana)
    """
    if len(df_normal) == 0:
        return df_normal.copy()
    
    # Convertir ventana de horas a días (MJD está en días)
    window_days = window_hours / 24.0
    
    # Ordenar por MJD y resetear índices
    df_sorted = df_normal.sort_values('MJD').copy().reset_index(drop=True)
    
    # Calcular flujo y error de flujo para propagación de errores
    mag_sorted = df_sorted['MAG'].values
    mag_err_sorted = df_sorted['MAGERR'].values
    flux_sorted = 10**(-mag_sorted / 2.5)
    flux_err_sorted = (mag_err_sorted * flux_sorted) / 1.086
    
    # Agrupar puntos en ventanas de tiempo
    grouped_data = []
    i = 0
    
    while i < len(df_sorted):
        # Inicio de la ventana
        window_start = df_sorted.iloc[i]['MJD']
        window_end = window_start + window_days
        
        # Encontrar todos los puntos dentro de esta ventana (usando índices del DataFrame ordenado)
        mask = (df_sorted['MJD'] >= window_start) & (df_sorted['MJD'] < window_end)
        window_indices = np.where(mask)[0]  # Índices numéricos (0, 1, 2, ...)
        
        if len(window_indices) == 0:
            i += 1
            continue
        
        # Si hay un solo punto, mantenerlo tal cual
        if len(window_indices) == 1:
            idx = window_indices[0]
            grouped_data.append({
                'MJD': df_sorted.iloc[idx]['MJD'],
                'MAG': df_sorted.iloc[idx]['MAG'],
                'MAGERR': df_sorted.iloc[idx]['MAGERR']
            })
        else:
            # Múltiples puntos en la ventana: calcular mediana
            flux_window = flux_sorted[window_indices]
            mjd_window = df_sorted.iloc[window_indices]['MJD'].values
            
            # Calcular mediana de flujo
            median_flux = np.median(flux_window)
            
            # Calcular error de la mediana: 1.253 × σ/√n
            # σ es la desviación estándar de los flujos en la ventana
            n = len(flux_window)
            if n > 1:
                std_flux = np.std(flux_window, ddof=1)  # ddof=1 para muestra (n-1)
                median_flux_err = 1.253 * std_flux / np.sqrt(n)
                # Proteger contra errores muy pequeños o cero
                # Si std_flux es 0 (todos los puntos iguales), usar el error promedio de los puntos
                if median_flux_err <= 0 or not np.isfinite(median_flux_err):
                    median_flux_err = np.mean(flux_err_sorted[window_indices])
            else:
                median_flux_err = flux_err_sorted[window_indices[0]]
            
            # Asegurar que el error sea finito y positivo
            if not np.isfinite(median_flux_err) or median_flux_err <= 0:
                median_flux_err = np.mean(flux_err_sorted[window_indices])
                if median_flux_err <= 0:
                    # Fallback: usar 2% del flujo como error mínimo
                    median_flux_err = median_flux * 0.02
            
            # Convertir flujo mediano y su error de vuelta a magnitud
            median_mag = -2.5 * np.log10(median_flux)
            # Propagación de errores inversa: σ_m = (1.086 * σ_F) / F
            median_mag_err = (1.086 * median_flux_err) / median_flux
            
            # Asegurar que el error de magnitud sea finito y positivo
            if not np.isfinite(median_mag_err) or median_mag_err <= 0:
                # Fallback: usar el error promedio de magnitud de los puntos en la ventana
                median_mag_err = np.mean(mag_err_sorted[window_indices])
                if median_mag_err <= 0:
                    median_mag_err = 0.01  # Error mínimo de 0.01 mag
            
            # Usar el MJD del punto mediano (o el centro de la ventana si hay empate)
            median_mjd = np.median(mjd_window)
            
            grouped_data.append({
                'MJD': median_mjd,
                'MAG': median_mag,
                'MAGERR': median_mag_err
            })
        
        # Avanzar al siguiente punto fuera de esta ventana
        i = window_indices.max() + 1
        if i >= len(df_sorted):
            break
    
    # Crear DataFrame con los datos agrupados
    if len(grouped_data) == 0:
        return pd.DataFrame(columns=df_normal.columns)
    
    df_grouped = pd.DataFrame(grouped_data)
    return df_grouped

def mjd_to_phase(mjd: np.ndarray, reference_mjd: Optional[float] = None) -> np.ndarray:
    """
    Convertir MJD a fase relativa
    
    Si reference_mjd es None, usa el mínimo como referencia
    """
    if reference_mjd is None:
        reference_mjd = mjd.min()
    return mjd - reference_mjd

def prepare_lightcurve(df: pd.DataFrame, filter_name: str = None, 
                       max_days_after_peak: float = 300.0, 
                       max_days_before_peak: float = 50.0,
                       max_days_before_first_obs: float = 20.0) -> Dict:
    """
    Preparar curva de luz para ajuste
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columnas MJD, MAG, MAGERR, Upperlimit
    filter_name : str, optional
        Nombre del filtro
    max_days_after_peak : float, default=300.0
        Máximo número de días después del pico de flujo para incluir en el ajuste
        Las supernovas típicamente no duran más de 300 días después del pico
    max_days_before_peak : float, default=50.0
        Máximo número de días antes del pico para incluir (para capturar el rise)
    max_days_before_first_obs : float, default=20.0
        Máximo número de días antes de la primera observación para incluir upper limits
    
    Returns:
    --------
    dict con: phase, mag, mag_err, flux, flux_err, sn_name, filter_name, peak_phase
    """
    # Separar datos normales y upper limits
    df_normal = df[~df['Upperlimit']].copy()
    df_ul = df[df['Upperlimit']].copy()
    
    # PRIMER FILTRO: Aplicar ventana temporal de 8 horas a detecciones normales
    # Esto agrupa puntos muy cercanos en el tiempo y los reemplaza por la mediana
    df_normal = apply_time_window_filter(df_normal, window_hours=8.0)
    
    # El modelo tiene 6 parámetros (A, f, t0, t_rise, t_fall, gamma)
    # Necesitamos al menos 7 detecciones para tener un sistema determinado (n > p)
    # Esta verificación se hace DESPUÉS del filtro de ventana
    if len(df_normal) < 7:
        return None
    
    # Convertir MJD a fase (relativa al mínimo MJD de datos normales)
    phase_normal = mjd_to_phase(df_normal['MJD'].values)
    
    # Convertir magnitud a flujo para datos normales
    mag_normal = df_normal['MAG'].values
    mag_err_normal = df_normal['MAGERR'].values
    
    flux_normal = 10**(-mag_normal / 2.5)
    # Propagación de errores: σ_F = (F * σ_m) / 1.086
    flux_err_normal = (mag_err_normal * flux_normal) / 1.086
    
    # IMPORTANTE: Para evitar que objetos débiles tengan sobrepeso excesivo en el chi-cuadrado
    min_relative_error_flux = 0.02  # 2% mínimo de error relativo en flujo
    flux_err_min = flux_normal * min_relative_error_flux
    flux_err_normal = np.maximum(flux_err_normal, flux_err_min)
    
    # Identificar el pico de flujo máximo (mínimo de magnitud = máximo de flujo)
    peak_idx = np.argmax(flux_normal)
    peak_phase = phase_normal[peak_idx]
    
    # INCLUIR LOS 3 ÚLTIMOS UPPER LIMITS ANTES DEL PRIMER PUNTO DE OBSERVACIÓN
    # Esto ayuda a dar contexto sobre el flujo antes de la explosión
    # CONSTRAINT: Solo incluir upper limits que estén dentro del rango configurado antes de la primera observación
    first_observation_mjd = df_normal['MJD'].min()
    # max_days_before_first_obs viene como parámetro (por defecto desde config.py)
    
    # Filtrar upper limits que estén antes de la primera observación
    ul_before = df_ul[df_ul['MJD'] < first_observation_mjd].copy()
    
    if len(ul_before) > 0:
        # Filtrar por constraint: máximo 20 días antes de la primera observación
        ul_before = ul_before[ul_before['MJD'] >= (first_observation_mjd - max_days_before_first_obs)].copy()
        
        if len(ul_before) > 0:
            # Ordenar por MJD descendente (más recientes primero) y tomar los 3 últimos
            ul_before = ul_before.sort_values('MJD', ascending=False).head(3)
            # Convertir a fase usando la misma referencia que los datos normales
            reference_mjd = df_normal['MJD'].min()
            phase_ul_before = mjd_to_phase(ul_before['MJD'].values, reference_mjd=reference_mjd)
            mag_ul_before = ul_before['MAG'].values
            flux_ul_before = 10**(-mag_ul_before / 2.5)
        else:
            # No hay upper limits dentro del rango de 20 días
            phase_ul_before = np.array([])
            mag_ul_before = np.array([])
            flux_ul_before = np.array([])
    else:
        phase_ul_before = np.array([])
        mag_ul_before = np.array([])
        flux_ul_before = np.array([])
    
    # Combinar datos normales con upper limits seleccionados
    phase = np.concatenate([phase_normal, phase_ul_before]) if len(phase_ul_before) > 0 else phase_normal
    mag = np.concatenate([mag_normal, mag_ul_before]) if len(mag_ul_before) > 0 else mag_normal
    mag_err = np.concatenate([mag_err_normal, np.full(len(mag_ul_before), np.nan)]) if len(mag_ul_before) > 0 else mag_err_normal
    flux = np.concatenate([flux_normal, flux_ul_before]) if len(flux_ul_before) > 0 else flux_normal
    flux_err = np.concatenate([flux_err_normal, np.full(len(flux_ul_before), np.nan)]) if len(flux_ul_before) > 0 else flux_err_normal
    
    # Crear máscara para identificar upper limits
    is_upper_limit = np.concatenate([
        np.zeros(len(phase_normal), dtype=bool),
        np.ones(len(phase_ul_before), dtype=bool)
    ]) if len(phase_ul_before) > 0 else np.zeros(len(phase_normal), dtype=bool)
    
    # Filtrar datos: solo hasta max_days_after_peak días después del pico
    # Si max_days_before_peak es None, no filtrar por días antes del peak (incluir todos desde la primera detección)
    # PERO mantener:
    # 1. Los upper limits seleccionados (están antes del primer punto)
    # 2. La primera detección normal (siempre incluirla para tener contexto del inicio)
    if max_days_before_peak is None:
        # Sin límite antes del peak: incluir todos los datos desde la primera detección hasta max_days_after_peak
        mask = phase <= peak_phase + max_days_after_peak
    else:
        # Con límite antes del peak: aplicar el filtro normal
        mask = (phase >= peak_phase - max_days_before_peak) & (phase <= peak_phase + max_days_after_peak)
    
    # Los upper limits seleccionados siempre se incluyen (están antes del primer punto)
    if len(phase_ul_before) > 0:
        mask_ul = is_upper_limit
        mask = mask | mask_ul
    
    phase_filtered = phase[mask]
    mag_filtered = mag[mask]
    mag_err_filtered = mag_err[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = flux_err[mask]
    is_upper_limit_filtered = is_upper_limit[mask] if len(is_upper_limit) > 0 else np.zeros(len(phase_filtered), dtype=bool)
    
    # Verificar que aún tenemos suficientes datos después del filtro
    # (contar solo datos normales, no upper limits)
    n_normal_points = np.sum(~is_upper_limit_filtered)
    if n_normal_points < 5:
        # Si el filtro es muy restrictivo, usar todos los datos pero con advertencia
        phase_filtered = phase
        mag_filtered = mag
        mag_err_filtered = mag_err
        flux_filtered = flux
        flux_err_filtered = flux_err
        is_upper_limit_filtered = is_upper_limit
    
    # Verificar si había upper limits ANTES de combinarlos (para el plot)
    had_upper_limits_before_combining = len(phase_ul_before) > 0
    
    # Guardar la referencia MJD usada para este filtro y los MJD originales
    reference_mjd = df_normal['MJD'].min()
    
    # Reconstruir MJD original de los datos filtrados
    # phase_filtered es relativo a reference_mjd, así que MJD = phase + reference_mjd
    mjd_filtered = phase_filtered + reference_mjd
    
    return {
        'phase': phase_filtered,
        'mjd': mjd_filtered,  # MJD original para plotear
        'mag': mag_filtered,
        'mag_err': mag_err_filtered,
        'flux': flux_filtered,
        'flux_err': flux_err_filtered,
        'is_upper_limit': is_upper_limit_filtered,  # Nueva: máscara para upper limits
        'had_upper_limits': had_upper_limits_before_combining,  # Indica si había upper limits antes de combinar
        'filter': filter_name,
        'peak_phase': peak_phase,  # Información adicional sobre el pico
        'reference_mjd': reference_mjd  # MJD usado como referencia (fase 0) para este filtro
    }

def load_supernovas_from_csv(csv_path: str) -> List[str]:
    """
    Leer CSV de supernovas exitosas y extraer lista de nombres
    
    Parameters:
    -----------
    csv_path : str or Path
        Ruta al archivo CSV con supernovas exitosas
    
    Returns:
    --------
    list : Lista de nombres de supernovas (sin duplicados, preservando orden)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Buscar columna con nombres de supernovas
        # Puede ser 'supernova_name', 'sn_name', 'name', etc.
        possible_columns = ['supernova_name', 'sn_name', 'name', 'supernova']
        sn_column = None
        
        for col in possible_columns:
            if col in df.columns:
                sn_column = col
                break
        
        if sn_column is None:
            # Si no encontramos ninguna columna conocida, usar la primera
            if len(df.columns) > 0:
                sn_column = df.columns[0]
                print(f"[WARNING] No se encontró columna conocida, usando primera columna: {sn_column}")
            else:
                raise ValueError("CSV vacío o sin columnas")
        
        # Extraer nombres, eliminar duplicados pero preservar orden
        supernovas = df[sn_column].dropna().unique().tolist()
        
        print(f"[INFO] CSV leído: {len(supernovas)} supernovas encontradas en columna '{sn_column}'")
        
        return supernovas
    
    except Exception as e:
        raise ValueError(f"Error leyendo CSV: {e}")

