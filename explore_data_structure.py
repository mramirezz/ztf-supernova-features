"""
Script de Exploración de Estructura de Datos ZTF Literature
============================================================

Este script explora la estructura de los archivos .dat de fotometría
de supernovas de la literatura ZTF para entender el formato antes
de implementar la extracción de features.

Uso:
    python explore_data_structure.py [tipo_supernova]
    
Ejemplo:
    python explore_data_structure.py "SN Ia"
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Ruta base de los datos
BASE_DATA_PATH = r"G:\Mi unidad\Work\Universidad\Phd\paper2_ZTF\Photometry_ZTF_ST_Alerce"

def parse_photometry_file(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parsear un archivo de fotometría .dat y extraer datos por filtro
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo .dat
        
    Returns:
    --------
    dict : Diccionario con {filtro: DataFrame}
        Cada DataFrame tiene columnas: MJD, MAG, MAGERR, Upperlimit
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer nombre de supernova del header
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
        
        # Detectar inicio de nueva sección de filtro
        if line.startswith('# FILTER'):
            # Guardar datos del filtro anterior si existen
            if current_filter and current_data:
                df = pd.DataFrame(current_data, columns=['MJD', 'MAG', 'MAGERR', 'Upperlimit', 
                                                          'Instrument', 'Telescope', 'Source'])
                # Convertir tipos
                df['MJD'] = pd.to_numeric(df['MJD'], errors='coerce')
                df['MAG'] = pd.to_numeric(df['MAG'], errors='coerce')
                df['MAGERR'] = pd.to_numeric(df['MAGERR'], errors='coerce')
                df['Upperlimit'] = df['Upperlimit'].str.strip().str.upper() == 'T'
                # Eliminar columnas innecesarias
                df = df[['MJD', 'MAG', 'MAGERR', 'Upperlimit']]
                df = df.dropna(subset=['MJD', 'MAG'])
                filters_data[current_filter] = df
            
            # Iniciar nueva sección
            current_filter = line.split('FILTER')[1].strip()
            current_data = []
        
        # Parsear línea de datos (empezar con tab o espacio)
        elif current_filter and (line.startswith('\t') or (line and not line.startswith('#'))):
            # Intentar parsear como datos tab-separados
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    # Verificar que sean números válidos
                    mjd = float(parts[0].strip())
                    mag = float(parts[1].strip())
                    magerr = float(parts[2].strip())
                    upperlimit = parts[3].strip() if len(parts) > 3 else 'F'
                    instrument = parts[4].strip() if len(parts) > 4 else 'nan'
                    telescope = parts[5].strip() if len(parts) > 5 else 'nan'
                    source = parts[6].strip() if len(parts) > 6 else 'nan'
                    
                    current_data.append([mjd, mag, magerr, upperlimit, instrument, telescope, source])
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
        df = df[['MJD', 'MAG', 'MAGERR', 'Upperlimit']]
        df = df.dropna(subset=['MJD', 'MAG'])
        filters_data[current_filter] = df
    
    return filters_data, sn_name

def explore_supernova_type(sn_type: str, max_files: int = 5) -> None:
    """
    Explorar archivos de un tipo de supernova específico
    
    Parameters:
    -----------
    sn_type : str
        Tipo de supernova (nombre de carpeta)
    max_files : int
        Número máximo de archivos a explorar
    """
    type_path = Path(BASE_DATA_PATH) / sn_type
    
    if not type_path.exists():
        print(f"[ERROR] La carpeta '{sn_type}' no existe en:")
        print(f"   {BASE_DATA_PATH}")
        return
    
    # Listar archivos
    dat_files = list(type_path.glob("*_photometry.dat"))
    
    if not dat_files:
        print(f"[WARNING] No se encontraron archivos .dat en '{sn_type}'")
        return
    
    print(f"\n{'='*80}")
    print(f"EXPLORACION: {sn_type}")
    print(f"{'='*80}")
    print(f"Total de archivos encontrados: {len(dat_files)}")
    print(f"Explorando primeros {min(max_files, len(dat_files))} archivos...\n")
    
    # Estadísticas generales
    all_filters = set()
    total_points = 0
    files_with_data = 0
    
    for i, filepath in enumerate(dat_files[:max_files]):
        print(f"\n{'-'*80}")
        print(f"Archivo {i+1}/{min(max_files, len(dat_files))}: {filepath.name}")
        print(f"{'-'*80}")
        
        try:
            filters_data, sn_name = parse_photometry_file(str(filepath))
            
            if not filters_data:
                print("   [WARNING] No se pudieron extraer datos de este archivo")
                continue
            
            files_with_data += 1
            print(f"   [OK] Supernova: {sn_name}")
            print(f"   [OK] Filtros encontrados: {', '.join(filters_data.keys())}")
            
            for filter_name, df in filters_data.items():
                all_filters.add(filter_name)
                n_points = len(df)
                total_points += n_points
                
                if n_points > 0:
                    mjd_min = df['MJD'].min()
                    mjd_max = df['MJD'].max()
                    mag_min = df['MAG'].min()
                    mag_max = df['MAG'].max()
                    duration = mjd_max - mjd_min
                    
                    print(f"\n   Filtro '{filter_name}':")
                    print(f"      - Puntos: {n_points}")
                    print(f"      - MJD range: {mjd_min:.2f} - {mjd_max:.2f}")
                    print(f"      - Duracion: {duration:.1f} dias")
                    print(f"      - Magnitud range: {mag_min:.2f} - {mag_max:.2f}")
                    print(f"      - Limites superiores: {df['Upperlimit'].sum()}")
                    
                    # Mostrar primeros 3 puntos
                    print(f"      - Primeros puntos:")
                    for idx, row in df.head(3).iterrows():
                        print(f"        MJD={row['MJD']:.2f}, MAG={row['MAG']:.3f}+/-{row['MAGERR']:.3f}")
        
        except Exception as e:
            print(f"   [ERROR] Error procesando archivo: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen
    print(f"\n{'='*80}")
    print(f"RESUMEN DE EXPLORACION")
    print(f"{'='*80}")
    print(f"   - Archivos procesados: {files_with_data}/{min(max_files, len(dat_files))}")
    print(f"   - Total de puntos: {total_points}")
    print(f"   - Filtros encontrados: {', '.join(sorted(all_filters))}")
    print(f"   - Promedio puntos/archivo: {total_points/files_with_data:.1f}" if files_with_data > 0 else "")

def list_all_types() -> None:
    """Listar todos los tipos de supernovas disponibles"""
    base_path = Path(BASE_DATA_PATH)
    
    if not base_path.exists():
        print(f"[ERROR] La ruta base no existe: {BASE_DATA_PATH}")
        return
    
    print(f"\n{'='*80}")
    print(f"TIPOS DE SUPERNOVAS DISPONIBLES")
    print(f"{'='*80}\n")
    
    types = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    
    for i, sn_type in enumerate(types, 1):
        type_path = base_path / sn_type
        dat_files = list(type_path.glob("*_photometry.dat"))
        print(f"{i:2d}. {sn_type:30s} ({len(dat_files):4d} archivos)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Explorar tipo específico
        sn_type = sys.argv[1]
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        explore_supernova_type(sn_type, max_files)
    else:
        # Listar todos los tipos disponibles
        list_all_types()
        print("\n" + "="*80)
        print("Uso: python explore_data_structure.py [tipo_supernova] [max_files]")
        print("   Ejemplo: python explore_data_structure.py 'SN Ia' 10")
        print("="*80)

