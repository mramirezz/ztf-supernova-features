"""
Configuración para extracción de features con MCMC
"""
from pathlib import Path

# Rutas
BASE_DATA_PATH = Path(r"G:\Mi unidad\Work\Universidad\Phd\paper2_ZTF\Photometry_ZTF_ST_Alerce")
OUTPUT_DIR = Path(__file__).parent / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
FEATURES_DIR = OUTPUT_DIR / "features"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Crear directorios
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Configuración MCMC
MCMC_CONFIG = {
    "n_walkers": 50,          # Número de walkers
    "n_steps": 2000,          # Pasos de MCMC
    "burn_in": 500,           # Burn-in steps
    "n_threads": 1,           # Threads paralelos
    "random_seed": 42         # Semilla aleatoria para reproducibilidad (None = aleatorio)
}

# Configuración del modelo
MODEL_CONFIG = {
    "param_names": ["A", "f", "t0", "t_rise", "t_fall", "gamma"],
    "bounds": {
        "A": (1e-10, 1e-5),
        "f": (0.0, 1.0),
        "t0": (-50.0, 50.0),
        "t_rise": (1.0, 100.0),
        "t_fall": (1.0, 100.0),
        "gamma": (1.0, 100.0)
    }
}

# Configuración de gráficos
PLOT_CONFIG = {
    "dpi": 300,
    "format": "png",
    "figsize": (10, 6)
}

# Filtros a procesar (lista de filtros a extraer features)
# Si está vacía, procesa todos los filtros disponibles
FILTERS_TO_PROCESS = ['g', 'r']  # Ejemplo: procesar filtros g y r

# Configuración de filtrado de datos
DATA_FILTER_CONFIG = {
    "max_days_after_peak": 300.0,  # Máximo número de días después del pico de flujo para incluir
    "max_days_before_peak": 50.0   # Máximo número de días antes del pico para incluir (para capturar el rise)
}

