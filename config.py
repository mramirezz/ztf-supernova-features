"""
Configuración para extracción de features con MCMC
"""
from pathlib import Path

# Rutas
# Ruta relativa: datos en ztf_literature_features/Photometry_ZTF_ST_Alerce
# Para pruebas locales, usar ruta absoluta; para producción, usar ruta relativa
BASE_DATA_PATH = Path(__file__).parent / "Photometry_ZTF_ST_Alerce"
OUTPUT_DIR = Path(__file__).parent / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
FEATURES_DIR = OUTPUT_DIR / "features"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
DEBUG_PDF_DIR = OUTPUT_DIR / "debug_pdfs"

# Crear directorios
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
DEBUG_PDF_DIR.mkdir(exist_ok=True)

# Configuración MCMC
MCMC_CONFIG = {
    "n_walkers": 100,         # Número de walkers
    "n_steps": 5000,          # Pasos de MCMC
    "burn_in": 500,           # Burn-in steps
    "n_threads": 1,           # Threads paralelos
    "random_seed": 41         # Semilla aleatoria para reproducibilidad (None = aleatorio)
}

# Configuración del modelo
MODEL_CONFIG = {
    "param_names": ["A", "f", "t0", "t_rise", "t_fall", "gamma"],
    "bounds": {
        "A": (1e-10, 1e-5),      # Amplitud típica de supernovas
        "f": (0.0, 1.0),
        "t0": (-200.0, 50.0),    # Tiempo de referencia: -200 a +50 días
        "t_rise": (1.0, 100.0),  # Tiempo de subida: 1-100 días
        "t_fall": (1.0, 200.0),  # Tiempo de caída: 1-200 días
        "gamma": (1.0, 150.0)    # Gamma: 1-150 días (duración del plateau)
    }
}

# Configuración de gráficos (estilo A&A)
PLOT_CONFIG = {
    "dpi": 150,
    "format": "png",
    "figsize": (6, 5),  # Proporción casi cuadrada estilo A&A column width
}

# Filtros a procesar (lista de filtros a extraer features)
# Si está vacía, procesa todos los filtros disponibles
FILTERS_TO_PROCESS = ['g', 'r']  # Ejemplo: procesar filtros g y r

# Configuración de filtrado de datos
DATA_FILTER_CONFIG = {
    "max_days_after_peak": 300.0,  # Máximo número de días después del pico de flujo para incluir
    "max_days_before_peak": None,   # Sin límite de días antes del pico (incluir todos los datos desde la primera detección)
    "max_days_before_first_obs": 20.0  # Máximo número de días antes de la primera observación para incluir upper limits
}

