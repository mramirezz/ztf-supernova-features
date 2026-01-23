"""
Lector de archivos .dat de fotometría ZTF
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
np.seterr(all='ignore')
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
    
    # NUEVO FILTRO DE UPPER LIMITS: Barrido inteligente antes de la primera detección
    # 1. Buscar en ventana de 60 días antes de la primera detección
    # 2. Si el último UL antes de la primera detección es más brillante (menor magnitud) que la primera detección, descartarlo
    # 3. Continuar hacia atrás hasta encontrar el primer UL más débil (mayor magnitud) que la primera detección
    # 4. Si no hay UL en esos 60 días, descartar la curva
    first_observation_mjd = df_normal['MJD'].min()
    first_observation_idx = df_normal['MJD'].idxmin()
    first_observation_mag = df_normal.loc[first_observation_idx, 'MAG']
    first_observation_flux = 10**(-first_observation_mag / 2.5)
    
    # Ventana de búsqueda: 60 días antes de la primera detección
    window_days = 60.0
    ul_search_start = first_observation_mjd - window_days
    
    # Función auxiliar para validar subida inicial (rise) en los primeros puntos
    def validate_slopes(df_normal):
        """
        Validar si hay una subida marcada (rise) en los primeros puntos.
        Criterio: El primer punto debe ser más débil (mayor magnitud) que los siguientes 2-3 puntos,
        o la pendiente entre el punto 1 y 3 debe ser negativa y significativa (< -0.05 mag/día).
        
        Returns:
        --------
        tuple: (is_valid: bool, details: str)
            is_valid: True si pasa la validación, False si no
            details: String con detalles de por qué pasó o no pasó
        """
        if len(df_normal) < 3:
            return False, f"No hay suficientes puntos para validar pendientes (solo {len(df_normal)} puntos, mínimo 3 requerido)"
        
        # Ordenar por MJD y tomar los primeros 3 puntos
        df_sorted = df_normal.sort_values('MJD').head(3)
        mjds = df_sorted['MJD'].values
        mags = df_sorted['MAG'].values
        
        # Calcular todas las pendientes para información detallada
        slope_12 = (mags[1] - mags[0]) / (mjds[1] - mjds[0]) if (mjds[1] - mjds[0]) > 0 else 0
        slope_23 = (mags[2] - mags[1]) / (mjds[2] - mjds[1]) if (mjds[2] - mjds[1]) > 0 else 0
        slope_13 = (mags[2] - mags[0]) / (mjds[2] - mjds[0]) if (mjds[2] - mjds[0]) > 0 else 0
        
        # Criterio 1: Verificar que el primer punto es más débil que los siguientes
        # (mag[0] > mag[1] y mag[0] > mag[2] indica subida clara)
        first_weaker_than_second = mags[0] > mags[1]
        first_weaker_than_third = mags[0] > mags[2]
        
        # Criterio 2: Pendiente negativa significativa entre punto 1 y 3
        # (pendiente negativa = magnitud disminuye = brillo aumenta = subida)
        significant_negative_slope = slope_13 < -0.05
        
        # Construir detalles
        details_parts = []
        details_parts.append(f"Puntos: P1(mjd={mjds[0]:.1f}, mag={mags[0]:.2f}), P2(mjd={mjds[1]:.1f}, mag={mags[1]:.2f}), P3(mjd={mjds[2]:.1f}, mag={mags[2]:.2f})")
        details_parts.append(f"Pendientes: slope_12={slope_12:.4f} mag/día, slope_23={slope_23:.4f} mag/día, slope_13={slope_13:.4f} mag/día")
        
        # Verificar criterios
        criteria_details = []
        if first_weaker_than_second and first_weaker_than_third:
            criteria_details.append("✓ Criterio 1: Primer punto más débil que P2 y P3")
        else:
            if not first_weaker_than_second:
                criteria_details.append(f"✗ Criterio 1: P1(mag={mags[0]:.2f}) NO es más débil que P2(mag={mags[1]:.2f})")
            if not first_weaker_than_third:
                criteria_details.append(f"✗ Criterio 1: P1(mag={mags[0]:.2f}) NO es más débil que P3(mag={mags[2]:.2f})")
        
        if significant_negative_slope:
            criteria_details.append(f"✓ Criterio 2: Pendiente 1-3 negativa significativa (slope_13={slope_13:.4f} < -0.05 mag/día)")
        else:
            criteria_details.append(f"✗ Criterio 2: Pendiente 1-3 no es negativa significativa (slope_13={slope_13:.4f}, requiere < -0.05 mag/día)")
        
        details_parts.extend(criteria_details)
        
        # Si cumple al menos uno de los criterios principales, hay subida marcada
        # (el primer punto más débil que el segundo Y el tercero, O pendiente negativa significativa)
        has_clear_rise = (first_weaker_than_second and first_weaker_than_third) or significant_negative_slope
        
        details_str = " | ".join(details_parts)
        
        if has_clear_rise:
            return True, f"Validación de pendientes PASÓ: {details_str}"
        else:
            return False, f"Validación de pendientes FALLÓ: {details_str}"
    
    # Primero verificar si hay upper limits en absoluto antes de la primera observación
    ul_all_before = df_ul[df_ul['MJD'] < first_observation_mjd].copy()
    
    # Inicializar selected_ul_before como DataFrame vacío
    selected_ul_before = pd.DataFrame()
    ul_validation_failed = False
    ul_failure_reason = None
    
    if len(ul_all_before) == 0:
        # No hay ningún upper limit antes de la primera detección
        ul_validation_failed = True
        ul_failure_reason = "No hay upper limits antes de la primera detección"
    else:
        # Filtrar upper limits que estén antes de la primera observación y dentro de la ventana de 60 días
        ul_before = ul_all_before[(ul_all_before['MJD'] >= ul_search_start)].copy()
        
        if len(ul_before) == 0:
            # Hay upper limits pero ninguno dentro de la ventana de 60 días
            ul_validation_failed = True
            ul_failure_reason = "No hay upper limits en ventana de 60 días antes de la primera detección (hay UL más lejanos pero fuera de ventana)"
        else:
            # Ordenar por MJD descendente (más recientes primero, más cerca de la primera detección)
            ul_before = ul_before.sort_values('MJD', ascending=False)
            
            # Convertir magnitudes a flujos para comparación
            ul_before_flux = 10**(-ul_before['MAG'].values / 2.5)
            
            # Buscar el PRIMER upper limit que sea más débil que la primera detección
            # (mayor magnitud = menor flujo), pero que NO sea prácticamente simultáneo
            # a la primera detección. Este será el UL principal.
            #
            # Luego, una vez encontrado ese primer UL válido, también incluimos
            # los SIGUIENTES 2 upper limits en la serie temporal (hasta un máximo de 3
            # en total), sin exigirles que sean más débiles que la primera detección.
            #
            # De esta forma:
            #   - El criterio "más débil que la primera detección" y el corte en tiempo
            #     (>= 1 día antes) se aplican SOLO al primer UL.
            #   - Los 2 siguientes nos dan información extra de no-detección en la
            #     misma fase temprana, siguiendo la idea original de usar varios UL.
            min_delta_days = 1.0
            first_valid_idx = None
            for i, (idx, row) in enumerate(ul_before.iterrows()):
                ul_flux = ul_before_flux[i]
                ul_mjd = row['MJD']

                # Descartar ULs que estén demasiado cerca en tiempo de la primera detección
                # (misma noche / mismo día), ya que en la práctica son casi simultáneos
                # y pueden terminar pesando más que la detección.
                if (first_observation_mjd - ul_mjd) < min_delta_days:
                    continue

                # Si el UL es más débil (menor flujo) que la primera detección, es candidato
                if ul_flux < first_observation_flux:
                    first_valid_idx = i
                    break
            
            if first_valid_idx is None:
                # No hay upper limits válidos (todos son más brillantes que la primera detección)
                ul_validation_failed = True
                ul_failure_reason = "Todos los upper limits en ventana de 60 días son más brillantes que la primera detección"
            else:
                # Seleccionar hasta 3 upper limits comenzando en el primero válido
                # (el primero cumple todas las condiciones; los 2 siguientes solo deben
                # estar dentro de la ventana temporal ya filtrada).
                selected_ul_before = ul_before.iloc[first_valid_idx:first_valid_idx+3]
    
    # Si falló la validación de upper limits, validar con pendientes antes de rechazar
    if ul_validation_failed:
        slope_valid, slope_details = validate_slopes(df_normal)
        if slope_valid:
            # Pendientes válidas: continuar sin upper limits
            print(f"    [INFO] {ul_failure_reason}, pero {slope_details}")
            print(f"    [INFO] Continuando sin upper limits")
            selected_ul_before = pd.DataFrame()
        else:
            # Construir mensaje de error detallado
            detailed_reason = f"{ul_failure_reason}. {slope_details}"
            raise ValueError(detailed_reason)
    
    # NUEVO FILTRO DE UPPER LIMITS: Barrido inteligente después de la última detección
    # 1. Buscar en ventana de 60 días después de la última detección
    # 2. Solo considerar ULs que sean más débiles (menor flujo) que la última detección
    # 3. Tomar los primeros 3 que cumplan esta condición
    last_observation_mjd = df_normal['MJD'].max()
    last_observation_idx = df_normal['MJD'].idxmax()
    last_observation_mag = df_normal.loc[last_observation_idx, 'MAG']
    last_observation_flux = 10**(-last_observation_mag / 2.5)
    
    # Ventana de búsqueda: 60 días después de la última detección
    window_days_after = 60.0
    ul_search_end = last_observation_mjd + window_days_after
    
    # Filtrar upper limits que estén después de la última observación y dentro de la ventana de 60 días
    ul_after = df_ul[(df_ul['MJD'] > last_observation_mjd) & (df_ul['MJD'] <= ul_search_end)].copy()
    
    # Si hay upper limits después, filtrar solo los que sean más débiles que la última detección
    selected_ul_after = pd.DataFrame()  # Inicializar como DataFrame vacío
    if len(ul_after) > 0:
        # Ordenar por MJD ascendente (más antiguos primero, más cerca de la última detección)
        ul_after = ul_after.sort_values('MJD', ascending=True)
        
        # Convertir magnitudes a flujos para comparación
        ul_after_flux = 10**(-ul_after['MAG'].values / 2.5)
        
        # Filtrar solo los que sean más débiles (menor flujo) que la última detección
        # y tomar los primeros 3
        valid_ul_after = []
        for i, (idx, row) in enumerate(ul_after.iterrows()):
            ul_flux = ul_after_flux[i]
            if ul_flux < last_observation_flux:
                valid_ul_after.append(idx)
                if len(valid_ul_after) >= 3:
                    break
        
        if len(valid_ul_after) > 0:
            selected_ul_after = ul_after.loc[valid_ul_after]
    
    # Convertir a fase usando la misma referencia que los datos normales
    reference_mjd = df_normal['MJD'].min()
    phase_ul_before = mjd_to_phase(selected_ul_before['MJD'].values, reference_mjd=reference_mjd) if len(selected_ul_before) > 0 else np.array([])
    mag_ul_before = selected_ul_before['MAG'].values if len(selected_ul_before) > 0 else np.array([])
    flux_ul_before = 10**(-mag_ul_before / 2.5) if len(mag_ul_before) > 0 else np.array([])
    
    phase_ul_after = mjd_to_phase(selected_ul_after['MJD'].values, reference_mjd=reference_mjd) if len(selected_ul_after) > 0 else np.array([])
    mag_ul_after = selected_ul_after['MAG'].values if len(selected_ul_after) > 0 else np.array([])
    flux_ul_after = 10**(-mag_ul_after / 2.5) if len(mag_ul_after) > 0 else np.array([])
    
    # Combinar datos normales con upper limits seleccionados (antes y después)
    phase_ul_all = np.concatenate([phase_ul_before, phase_ul_after]) if len(phase_ul_before) > 0 or len(phase_ul_after) > 0 else np.array([])
    mag_ul_all = np.concatenate([mag_ul_before, mag_ul_after]) if len(mag_ul_before) > 0 or len(mag_ul_after) > 0 else np.array([])
    flux_ul_all = np.concatenate([flux_ul_before, flux_ul_after]) if len(flux_ul_before) > 0 or len(flux_ul_after) > 0 else np.array([])
    
    phase = np.concatenate([phase_normal, phase_ul_all]) if len(phase_ul_all) > 0 else phase_normal
    mag = np.concatenate([mag_normal, mag_ul_all]) if len(mag_ul_all) > 0 else mag_normal
    mag_err = np.concatenate([mag_err_normal, np.full(len(mag_ul_all), np.nan)]) if len(mag_ul_all) > 0 else mag_err_normal
    flux = np.concatenate([flux_normal, flux_ul_all]) if len(flux_ul_all) > 0 else flux_normal
    flux_err = np.concatenate([flux_err_normal, np.full(len(flux_ul_all), np.nan)]) if len(flux_ul_all) > 0 else flux_err_normal
    
    # Crear máscara para identificar upper limits
    n_ul = len(phase_ul_all)
    is_upper_limit = np.concatenate([
        np.zeros(len(phase_normal), dtype=bool),
        np.ones(n_ul, dtype=bool)
    ]) if n_ul > 0 else np.zeros(len(phase_normal), dtype=bool)
    
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
    had_upper_limits_before_combining = len(phase_ul_all) > 0
    
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
    
    # Si la ruta es relativa, intentar resolverla de varias formas
    original_path = csv_path
    if not csv_path.is_absolute():
        # 1. Intentar como ruta relativa al directorio de trabajo actual
        if not csv_path.exists():
            # 2. Intentar como ruta relativa al directorio del script
            script_dir = Path(__file__).parent
            potential_path = script_dir / csv_path
            if potential_path.exists():
                csv_path = potential_path
            # 3. Si aún no existe, mantener el path original para el mensaje de error
    
    if not csv_path.exists():
        # Intentar sugerir archivos similares en el mismo directorio
        error_msg = f"Archivo CSV no encontrado: {original_path}"
        if not original_path.is_absolute():
            # Buscar archivos similares en el directorio del script
            script_dir = Path(__file__).parent
            csv_dir = script_dir / original_path.parent if original_path.parent != Path('.') else script_dir
            if csv_dir.exists() and csv_dir.is_dir():
                # Buscar archivos CSV en ese directorio
                csv_files = list(csv_dir.glob("*.csv"))
                if csv_files:
                    error_msg += f"\n[INFO] Archivos CSV encontrados en {csv_dir}:"
                    for f in csv_files[:5]:  # Mostrar máximo 5
                        error_msg += f"\n  - {f.name}"
                    if len(csv_files) > 5:
                        error_msg += f"\n  ... y {len(csv_files) - 5} más"
        raise FileNotFoundError(error_msg)
    
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

