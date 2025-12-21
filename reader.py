"""
Lector de archivos .dat de fotometría ZTF
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

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
                       max_days_before_peak: float = 50.0) -> Dict:
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
    
    Returns:
    --------
    dict con: phase, mag, mag_err, flux, flux_err, sn_name, filter_name, peak_phase
    """
    # Filtrar límites superiores si es necesario
    df_clean = df[~df['Upperlimit']].copy()
    
    if len(df_clean) < 5:
        return None
    
    # Convertir MJD a fase (relativa al mínimo MJD)
    phase = mjd_to_phase(df_clean['MJD'].values)
    
    # Convertir magnitud a flujo
    mag = df_clean['MAG'].values
    mag_err = df_clean['MAGERR'].values
    
    flux = 10**(-mag / 2.5)
    flux_err = (mag_err * flux) / 1.086
    
    # Identificar el pico de flujo máximo (mínimo de magnitud = máximo de flujo)
    peak_idx = np.argmax(flux)
    peak_phase = phase[peak_idx]
    
    # Filtrar datos: solo hasta max_days_after_peak días después del pico
    # y hasta max_days_before_peak días antes del pico
    mask = (phase >= peak_phase - max_days_before_peak) & (phase <= peak_phase + max_days_after_peak)
    
    phase_filtered = phase[mask]
    mag_filtered = mag[mask]
    mag_err_filtered = mag_err[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = flux_err[mask]
    
    # Verificar que aún tenemos suficientes datos después del filtro
    if len(phase_filtered) < 5:
        # Si el filtro es muy restrictivo, usar todos los datos pero con advertencia
        phase_filtered = phase
        mag_filtered = mag
        mag_err_filtered = mag_err
        flux_filtered = flux
        flux_err_filtered = flux_err
    
    return {
        'phase': phase_filtered,
        'mag': mag_filtered,
        'mag_err': mag_err_filtered,
        'flux': flux_filtered,
        'flux_err': flux_err_filtered,
        'filter': filter_name,
        'peak_phase': peak_phase  # Información adicional sobre el pico
    }

