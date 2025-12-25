# Extracción de Features de Supernovas ZTF - Literatura

## Resumen del Proyecto

Este módulo está diseñado para extraer features del modelo de Villar a partir de datos de fotometría de supernovas de la literatura ZTF almacenados en formato `.dat`. Utiliza MCMC (Markov Chain Monte Carlo) con `emcee` para ajustar el modelo ALERCE y extraer 25 características de cada curva de luz.

## Configuración del Entorno

### Crear Entorno Conda

```bash
# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate ztf_features/streamlit_env
```

### Verificar Instalación

```bash
# Verificar que el entorno está activo
conda info --envs

# Verificar paquetes instalados
conda list
```

**Nota**: El entorno se llama `ztf_features` y contiene todas las dependencias necesarias (numpy, pandas, matplotlib, scipy, emcee, corner, streamlit).

## Estructura del Proyecto

```
ztf_literature_features/
├── README.md                    # Este archivo
├── environment.yml              # Entorno conda (RECOMENDADO)
├── requirements.txt             # Dependencias pip (alternativa)
├── config.py                   # Configuración (rutas, parámetros MCMC)
├── reader.py                   # Lector de archivos .dat
├── model.py                    # Modelo de Villar (ALERCE)
├── mcmc_fitter.py              # Ajuste con MCMC (emcee)
├── feature_extractor.py        # Extracción de features
├── plotter.py                  # Generación de gráficos
├── main.py                     # Script principal (batch processing)
├── streamlit_app.py            # App Streamlit para exploración interactiva
├── explore_data_structure.py   # Script de exploración de datos
    └── outputs/                    # Directorio de salida
    ├── plots/                  # Gráficos guardados (organizados por tipo y supernova)
    │   └── {sn_type}/         # Subcarpeta por tipo de supernova
    │       └── {sn_name}/     # Subcarpeta por supernova
    │           ├── {sn_name}_{filter}_fit.png
    │           └── {sn_name}_{filter}_corner.png
    ├── features/               # Features extraídas (CSV)
    │   └── features_{sn_type}.csv
    ├── checkpoints/            # Archivos de checkpoint para retomar procesamiento
    │   └── checkpoint_{sn_type}.json
    ├── logs/                   # Archivos de log
    │   └── log_{sn_type}_{timestamp}.log
    └── debug_pdfs/             # PDFs de debug (modo --debug-pdf)
        ├── SN_Ia_debug.pdf
        ├── SN_Ia_successful.csv
        └── ...
```

## Datos Fuente

**Ubicación**: Los datos deben estar en `ztf_literature_features/Photometry_ZTF_ST_Alerce/`

El código lee desde una ruta relativa: `{directorio_del_proyecto}/ztf_literature_features/Photometry_ZTF_ST_Alerce/`

**Estructura esperada**:
```
ztf_literature_features/
└── Photometry_ZTF_ST_Alerce/
    ├── SN Ia/
    │   ├── ZTF18aasufva_photometry.dat
    │   ├── ZTF20acgnrqu_photometry.dat
    │   └── ...
    ├── SN II/
    │   └── ...
    └── ...
```

### Características de los Datos

- **Formato**: Archivos `.dat` con múltiples filtros por archivo
- **Estructura**: 
  - Header con nombre de supernova
  - Secciones separadas por filtro (g, r, i, etc.)
  - Columnas: MJD, MAG, MAGERR, Upperlimit
- **Organización**: 29 tipos de supernovas en carpetas separadas
- **Total**: Miles de archivos (ej: ~6000 archivos en "SN Ia")

### Tipos de Supernovas Disponibles (29 tipos)

| # | Tipo | Cantidad de Archivos | Descripción |
|---|------|---------------------|-------------|
| 1 | **SN Ia** | 5,992 | Supernovas tipo Ia estándar |
| 2 | **SN II** | 1,234 | Tipo II genérico |
| 3 | **SN IIn** | 234 | Tipo IIn |
| 4 | **SN Ia-91T-like** | 193 | Subtipo Ia 91T |
| 5 | **SN Ic** | 176 | Tipo Ic estándar |
| 6 | **SN Ib** | 135 | Tipo Ib estándar |
| 7 | **SN IIb** | 133 | Tipo IIb |
| 8 | **SLSN-I** | 109 | Superluminosas tipo I |
| 9 | **SN IIP** | 110 | Tipo IIP (plateau) |
| 10 | **SLSN-II** | 68 | Superluminosas tipo II |
| 11 | **SN Ic-BL** | 78 | Ic broad-line |
| 12 | **SN Ia-91bg-like** | 51 | Subtipo Ia 91bg |
| 13 | **SN Ia-pec** | 50 | Ia peculiares |
| 14 | **SN Ibc** | 32 | Tipo Ibc (combinado) |
| 15 | **SN Ibn** | 30 | Tipo Ibn |
| 16 | **SN Iax[02cx-like]** | 25 | Tipo Iax |
| 17 | **SN I** | 25 | Tipo I genérico |
| 18 | **SN** | 25 | Sin clasificación específica |
| 19 | **SN Ia-CSM** | 21 | Ia con material circumestelar |
| 20 | **SN Ia-SC** | 13 | Ia super-Chandrasekhar |
| 21 | **SN II-pec** | 6 | II peculiares |
| 22 | **SN Ib-pec** | 6 | Ib peculiares |
| 23 | **SN Icn** | 3 | Tipo Icn |
| 24 | **SN Ib-Ca-rich** | 3 | Ib ricas en calcio |
| 25 | **SN IIn-pec** | 2 | IIn peculiares |
| 26 | **SN Ic-pec** | 1 | Ic peculiares |
| 27 | **SN IIL** | 1 | Tipo II lineal |
| 28 | **SN Ia-Ca-rich** | 1 | Ia ricas en calcio |
| 29 | **SN Ic-Ca-rich** | 0 | Ic ricas en calcio |

**Total**: ~8,200 archivos de fotometría

**Nota**: Los números pueden variar ligeramente si se agregan nuevos archivos. Para verificar los números actuales, ejecuta:
```bash
python explore_data_structure.py
```

## Formato de los Archivos .dat

### Estructura del archivo:

```
###################### HEADER ######################
# SNNAME:	ZTF18aasufva
###################### HEADER ######################

# FILTER g
#        MJD  MAG  MAGERR  Upperlimit  Instrument  Telescope  Source
	58280.217	18.641	0.041	F	nan	nan	nan
	58283.242	18.539	0.042	F	nan	nan	nan
	...

# FILTER r
#        MJD  MAG  MAGERR  Upperlimit  Instrument  Telescope  Source
	58245.152	20.032	0.116	F	nan	nan	nan
	...
```

### Características del formato:

- **Nombre del archivo**: `{ZTF_ID}_photometry.dat`
- **Header**: Contiene el nombre de la supernova
- **Múltiples filtros**: Cada archivo puede contener datos de varios filtros (g, r, i, etc.)
- **Columnas por filtro**:
  - `MJD`: Modified Julian Date (tiempo)
  - `MAG`: Magnitud
  - `MAGERR`: Error en la magnitud
  - `Upperlimit`: Flag (F = False, T = True)
  - `Instrument`, `Telescope`, `Source`: Generalmente `nan`
- **Separador**: Tabs (`\t`)

## Uso del Script de Exploración

**Propósito**: El script `explore_data_structure.py` permite explorar y entender la estructura de los datos antes de procesarlos. Útil para:
- Ver qué tipos de supernovas están disponibles
- Verificar la cantidad de archivos por tipo
- Inspeccionar el formato de archivos individuales
- Entender qué filtros están disponibles en cada archivo
- Validar la estructura de datos antes del procesamiento masivo

**Ubicación de datos**: El script `explore_data_structure.py` tiene su propia ruta hardcodeada (puede diferir de `config.py`). El procesamiento principal (`main.py`, `streamlit_app.py`) lee desde:
- **Ruta relativa**: `ztf_literature_features/Photometry_ZTF_ST_Alerce/` (definida en `config.py`)
- **Estructura**: `{BASE_DATA_PATH}/{tipo_supernova}/*_photometry.dat`
- **Ejemplo**: `ztf_literature_features/Photometry_ZTF_ST_Alerce/SN Ia/ZTF18aasufva_photometry.dat`

**Nota**: Asegúrate de que la carpeta `Photometry_ZTF_ST_Alerce` esté dentro de `ztf_literature_features/` para que el código funcione correctamente.

### Listar todos los tipos disponibles:
```bash
python explore_data_structure.py
```
Muestra una tabla con todos los tipos de supernovas y la cantidad de archivos de cada tipo.

### Explorar un tipo específico:
```bash
python explore_data_structure.py "SN Ia" 5
```
Muestra información detallada de hasta 5 archivos del tipo especificado, incluyendo:
- Nombre de cada supernova
- Filtros disponibles en cada archivo
- Número de puntos de datos por filtro
- Rango de fechas (MJD)
- Ejemplo de datos

**Parámetros**:
- `tipo_supernova`: Nombre exacto de la carpeta (ej: "SN Ia", "SLSN-II")
- `max_files`: Número máximo de archivos a explorar (opcional, default: 5)

## Instalación y Uso

### Crear Entorno Conda (Recomendado)

```bash
# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate ztf_features
```

### Instalación con pip (Alternativa)

Si prefieres usar pip directamente:

```bash
pip install -r requirements.txt
```

## Procesamiento Batch (main.py)

**Para procesar múltiples supernovas y generar features en CSV:**

```bash
# Activar entorno
conda activate ztf_features

# Procesar 3 supernovas tipo Ia con filtros g y r
python main.py "SN Ia" 3 "g,r"
```

**Características:**
- Procesa múltiples supernovas automáticamente
- Genera CSV con todas las features
- Guarda gráficos y corner plots para cada filtro
- **Sistema de checkpoint**: Permite interrumpir y retomar desde donde quedó
- **Reproducibilidad**: Usa semilla fija (configurable en `config.py`)
- **Optimización de memoria**: Libera memoria después de cada supernova para procesar miles sin problemas
- Ideal para procesamiento masivo

**Parámetros:**
1. **Tipo de supernova** (requerido): Nombre exacto de la carpeta (ej: "SN Ia", "SLSN-II")
2. **Número de supernovas** (requerido): 
   - Número entero (ej: `3`) para procesar las primeras N
   - `all` para procesar todas las supernovas disponibles
3. **Filtros a procesar** (opcional): 
   - Formato: `"g,r"` o `"g r"` o `"['g','r']"`
   - Si no especificas, usa `FILTERS_TO_PROCESS` de `config.py` (por defecto: `['g', 'r']`)
   - Si `FILTERS_TO_PROCESS = []`, procesa todos los filtros disponibles
4. **`--resume`** (opcional): 
   - Flag para continuar desde checkpoint
   - Salta automáticamente las supernovas/filtros ya procesados
5. **`--debug-pdf`** (opcional): 
   - Modo especial para generar PDFs de debug con múltiples supernovas
   - Genera un PDF con fit plots y corner plots para inspección visual
   - Ver sección "Modo Debug PDF" para más detalles

**Ejemplos de Uso:**

```bash
# SOBRESCRIBE: Ejecución básica (3 supernovas, filtros por defecto)
# Procesa todo desde cero y SOBRESCRIBE archivos existentes
python main.py "SN Ia" 3

# SOBRESCRIBE: Con filtros específicos
# Procesa y SOBRESCRIBE archivos existentes
python main.py "SN Ia" 3 "g,r"

# SOBRESCRIBE: Con espacios en filtros
# Procesa y SOBRESCRIBE archivos existentes
python main.py "SN Ia" 3 "g r"

# SOBRESCRIBE: Procesar TODAS las supernovas
# Procesa todas y SOBRESCRIBE archivos existentes
python main.py "SN Ia" all

# NO SOBRESCRIBE: Continuar desde checkpoint (retoma donde quedó)
# Salta automáticamente las combinaciones ya procesadas, NO sobrescribe
python main.py "SN Ia" 3 --resume

# NO SOBRESCRIBE: Procesar todas con checkpoint
# Salta automáticamente las combinaciones ya procesadas, NO sobrescribe
python main.py "SN Ia" all --resume

# NO SOBRESCRIBE: Combinar filtros y checkpoint
# Salta automáticamente las combinaciones ya procesadas, NO sobrescribe
python main.py "SN Ia" 10 "g,r,i" --resume
```

**Importante - Comportamiento de Sobrescritura:**

- **Sin `--resume`**: Todos los comandos **SOBRESCRIBEN** archivos existentes (gráficos, features, etc.). Procesa todo desde cero.
- **Con `--resume`**: Todos los comandos **NO SOBRESCRIBEN**. Salta automáticamente las combinaciones (supernova + filtro) que ya están en el checkpoint y solo procesa las nuevas.

**Sistema de Checkpoint:**

El sistema de checkpoint permite interrumpir y retomar el procesamiento sin perder trabajo:

- **Archivo de checkpoint**: Se guarda en `outputs/checkpoints/checkpoint_{tipo_sn}.json`
- **Actualización automática**: Se actualiza después de procesar cada filtro exitosamente
- **Sin duplicados**: Si usas `--resume`, salta automáticamente las combinaciones ya procesadas
- **Formato del checkpoint**:
  ```json
  {
    "sn_type": "SN Ia",
    "processed": [
      ["ZTF20acgnrqu", "g"],
      ["ZTF20acgnrqu", "r"],
      ["ZTF20acggxac", "g"]
    ],
    "last_updated": "2024-01-15 14:30:25"
  }
  ```

**Ventajas del Checkpoint:**
- Puedes interrumpir el proceso (Ctrl+C) y retomar después
- No procesa duplicados (ahorra tiempo)
- Guarda progreso incremental (no pierdes trabajo)
- Compatible con ejecuciones anteriores (sin `--resume` funciona igual)

**Reproducibilidad:**

- **Semilla aleatoria**: Configurada en `config.py` (`MCMC_CONFIG["random_seed"] = 42`)
- **Resultados consistentes**: Con la misma semilla, los resultados MCMC son reproducibles
- **Mismo ajuste**: Si ejecutas dos veces con los mismos datos y parámetros, obtienes el mismo resultado

**Optimización de Memoria:**

El código está optimizado para procesar miles de supernovas sin quedarse sin memoria:

- **Liberación de figuras**: Las figuras de matplotlib se cierran inmediatamente después de guardarlas
- **Eliminación de samples MCMC**: Los samples grandes se eliminan después de generar los plots
- **Recolección de basura**: `gc.collect()` se ejecuta periódicamente (después de cada filtro, cada supernova, y cada 10 supernovas)
- **Sin acumulación**: Los datos se procesan y liberan inmediatamente, no se acumulan en memoria

**Nota**: Cada filtro genera un registro separado en el CSV. Si procesas 3 supernovas con 2 filtros cada una, obtendrás 6 registros en total.

## Modo Debug PDF (--debug-pdf)

**Para generar PDFs de inspección visual con múltiples supernovas:**

```bash
# Generar PDF con 200 supernovas tipo Ia (por defecto para SN Ia)
python main.py "SN Ia" --debug-pdf

# Especificar número de supernovas
python main.py "SN Ia" 50 --debug-pdf

# Con filtros específicos
python main.py "SN Ia" 100 "g,r" --debug-pdf

# Para otros tipos (valores por defecto: 50 para subtipos, 100 para otros)
python main.py "SN Ia-91bg-like" --debug-pdf
python main.py "SN Ia-91T-like" --debug-pdf
```

**Características del Modo Debug:**

- **Selección aleatoria**: Selecciona supernovas aleatoriamente (no solo las primeras N)
- **Filtrado por año**: Solo procesa supernovas del año 2022 en adelante (ZTF22, ZTF23, etc.)
- **Validación**: Solo incluye supernovas con al menos 6 detecciones normales (excluyendo upper limits)
- **Continuación automática**: Continúa intentando hasta obtener el número solicitado de supernovas exitosas
- **Una página por supernova**: Cada página contiene:
  - Si hay 1 filtro: Fit plot (magnitud y flujo) arriba, corner plot abajo
  - Si hay 2 filtros: Fit plot filtro 1, fit plot filtro 2, corner plot (del filtro 1)
- **Rango común de X**: Cuando hay 2 filtros, ambos comparten el mismo rango de MJD para facilitar comparación
- **Eje X en MJD**: Los plots usan fechas originales (MJD) en lugar de fase relativa
- **CSV de supernovas exitosas**: Genera un CSV adicional con la lista de supernovas procesadas exitosamente
- **Corner plot mejorado**: Muestra números pequeños en notación científica cuando es necesario (ej: A = 1.23e-08 en lugar de 0.00)

**Archivos generados:**

- **PDF**: `outputs/debug_pdfs/{sn_type}_debug.pdf`
- **CSV**: `outputs/debug_pdfs/{sn_type}_successful.csv` (lista de supernovas exitosas)

**Valores por defecto según tipo:**

- **SN Ia**: 200 supernovas
- **SN Ia-91bg-like, SN Ia-91T-like**: 50 supernovas
- **Otros tipos**: 100 supernovas

**Notas:**

- El modo debug NO guarda features en el CSV principal (solo genera PDFs)
- El modo debug NO usa checkpoint (siempre procesa desde cero)
- El modo debug NO sobrescribe PDFs existentes (genera nuevos cada vez)
- Los plots en el PDF usan el mismo estilo y funciones que el procesamiento normal

## Exploración Interactiva (streamlit_app.py)

**Para explorar y hacer sanity checks de supernovas individuales:**

```bash
# Activar entorno
conda activate ztf_features

# Ejecutar app interactiva
streamlit run streamlit_app.py
```

**Características:**
- Interfaz web interactiva
- Selecciona supernova específica desde dropdown
- Ejecuta MCMC en tiempo real (no necesita ejecutar main.py primero)
- Visualiza ajustes y corner plots inmediatamente
- Ajusta parámetros MCMC interactivamente
- **Visualización de distribuciones**: Tab dedicado para ver distribuciones de parámetros desde el CSV (sin necesidad de ejecutar MCMC)
- **Opción opcional para guardar**: Puedes activar "Guardar gráficos y features" en la barra lateral
  - Si activas: Guarda gráficos, corner plots y CSV con features
  - Si no activas: Solo muestra (no guarda nada, no sobreescribe)
- Ideal para exploración y validación visual

**Funcionalidades del Streamlit App:**

1. **Selección de Supernova**: Dropdown para elegir tipo y archivo específico
2. **Configuración MCMC**: Sliders para ajustar walkers, steps, burn-in
3. **Selección de Filtros**: Checkboxes para seleccionar múltiples filtros simultáneamente
4. **Vista Completa**: Gráfico de contexto mostrando todos los datos (incluyendo upper limits)
5. **Resultados MCMC**: Métricas de los 6 parámetros principales con descripciones
6. **Gráficos de Ajuste**: 
   - Dos subplots (magnitud y flujo) compartiendo eje X
   - Eje X en MJD (fechas originales) para mejor contexto temporal
   - Sin espacio vertical entre subplots para layout compacto
   - Muestra mediana, promedio y múltiples realizaciones MCMC
   - Estilo profesional para papers
7. **Corner Plot**: Distribución de parámetros MCMC con formato mejorado para números pequeños (notación científica cuando es necesario)
8. **Distribuciones de Parámetros**: Tab que muestra histogramas de los 6 parámetros principales separados por filtro, basados en datos del CSV (disponible sin ejecutar MCMC)

**Diferencias con main.py:**
- `main.py`: Procesamiento batch de múltiples supernovas → genera CSV consolidado
- `streamlit_app.py`: Exploración interactiva de una supernova → visualización inmediata

**Compatibilidad y Consistencia:**
- **Mismos resultados**: Si procesas la misma supernova con los mismos parámetros MCMC, ambos deberían dar resultados idénticos
- **Mismo formato CSV**: Ambos guardan en `features_{sn_type}.csv` (consolidado por tipo)
- **Mismos gráficos**: Ambos guardan `{sn_name}_{filter_name}_fit.png` y `{sn_name}_{filter_name}_corner.png`
- **Mismos parámetros por defecto**: Streamlit usa los valores de `config.py` como valores iniciales
- **Nota**: Streamlit permite cambiar parámetros MCMC interactivamente, lo que puede dar resultados diferentes

**Puedes usar cualquiera de los dos**, dependiendo de tu necesidad:
- **Batch processing**: Usa `main.py` para procesar muchas supernovas
- **Exploración/validación**: Usa `streamlit_app.py` para revisar ajustes específicos
- **Ambos pueden coexistir**: Puedes procesar algunas con `main.py` y otras con `streamlit_app.py`, todas se guardan en el mismo CSV consolidado

## Flujo de Trabajo

1. **reader.py**: Lee archivo .dat → parsea filtros → convierte MJD a fase → convierte magnitud a flujo
2. **model.py**: Define modelo de Villar (ALERCE) y funciones de conversión
3. **mcmc_fitter.py**: Ajusta con MCMC usando emcee (en espacio de flujo, con semilla fija para reproducibilidad)
4. **feature_extractor.py**: Extrae las 25 features
5. **plotter.py**: Genera gráficos (ajuste + corner plot) con estilo profesional para papers
6. **main.py**: Orquesta todo para procesar múltiples supernovas
   - Guarda checkpoint después de cada filtro procesado
   - Permite retomar desde checkpoint con `--resume`
   - Libera memoria después de cada supernova

## Features Extraídas (25 totales)

- **Parámetros principales** (6): A, f, t0, t_rise, t_fall, gamma
- **Errores formales** (6): A_err, f_err, t0_err, t_rise_err, t_fall_err, gamma_err
- **Errores MCMC** (6): A_mc_std, f_mc_std, t0_mc_std, t_rise_mc_std, t_fall_mc_std, gamma_mc_std
- **Métricas de ajuste** (3): rms, mad, median_relative_error_pct
- **Características de curva** (2): n_points, time_span
- **Metadatos** (2): sn_name, filter_band

## Configuración

### Parámetros MCMC (config.py)

Los parámetros por defecto están en `config.py`:

```python
MCMC_CONFIG = {
    "n_walkers": 50,
    "n_steps": 2000,
    "burn_in": 500,
    "random_seed": 42  # Para reproducibilidad
}
```

### Filtrado de Datos

El código filtra automáticamente los datos para enfocarse en la fase relevante de la supernova:

```python
DATA_FILTER_CONFIG = {
    "max_days_after_peak": 300.0,   # Máximo 300 días después del pico
    "max_days_before_peak": 50.0    # Máximo 50 días antes del pico
}
```

Si después del filtro quedan menos de 6 puntos de detección normal (excluyendo upper limits), la supernova se omite del procesamiento.

### Estilo de Gráficos

Los gráficos están configurados con estilo profesional para papers:

- **Fuente**: DejaVu Sans (sans-serif, compatible con caracteres especiales)
- **Idioma**: Inglés
- **Formato**: 
  - Eje X compartido entre subplots de magnitud y flujo
  - Eje X en MJD (fechas originales) cuando se procesa con datos completos
  - Sin espacio vertical entre subplots (hspace=0.0) para layout compacto
  - Título único para toda la figura
  - Ticks hacia adentro
  - Colores representativos para filtros (verde para g, rojo para r)
  - Nombres de filtros en minúscula (g, r) para filtros Sloan
  - Cuando hay 2 filtros en modo debug, ambos comparten el mismo rango de MJD para facilitar comparación

## Guardar Resultados

### Comportamiento de Guardado

| Script | Gráficos | Features CSV | Checkpoint | Log |
|--------|----------|--------------|------------|-----|
| `main.py` | Siempre guarda | Siempre guarda | Siempre guarda | Siempre guarda |
| `streamlit_app.py` | Solo si activas "Guardar" | Solo si activas "Guardar" | No guarda | No guarda |

**Nota importante**: Ambos scripts guardan en los mismos archivos y directorios. Si procesas la misma supernova con ambos, el último en ejecutarse sobrescribirá los resultados.

### Organización de Archivos

- **Gráficos**: `outputs/plots/{sn_type}/{sn_name}/{sn_name}_{filter}_fit.png` y `{sn_name}_{filter}_corner.png`
- **Features**: `outputs/features/features_{sn_type}.csv` (consolidado por tipo)
- **Checkpoints**: `outputs/checkpoints/checkpoint_{sn_type}.json`
- **Logs**: `outputs/logs/log_{sn_type}_{timestamp}.log`

### Formato del CSV de Features

El CSV contiene una fila por combinación (supernova, filtro). Si una combinación ya existe, se reemplaza (no se duplica).

Columnas principales:
- `sn_name`: Nombre de la supernova
- `filter_band`: Filtro (g, r, etc.)
- `A`, `f`, `t0`, `t_rise`, `t_fall`, `gamma`: Parámetros del modelo
- `A_err`, `f_err`, ...: Errores formales
- `A_mc_std`, `f_mc_std`, ...: Desviaciones estándar MCMC
- `rms`, `mad`, `median_relative_error_pct`: Métricas de ajuste
- `n_points`, `time_span`: Características de la curva
- `sn_type`: Tipo de supernova

## Logging

El script `main.py` genera archivos de log detallados en `outputs/logs/`:

- **Formato**: `log_{sn_type}_{timestamp}.log`
- **Contenido**: 
  - Progreso de procesamiento
  - Tiempos de ejecución
  - Errores y warnings
  - Separadores claros entre supernovas
- **Sin emojis**: Los logs están diseñados para ser legibles y profesionales

## Notas Técnicas

### MCMC en Espacio de Flujo

El ajuste MCMC se realiza en el espacio de flujo (no en magnitud) porque:
- El modelo de Villar está definido en flujo
- Los errores son más gaussianos en flujo
- Mejor convergencia del MCMC

Los resultados se convierten a magnitud solo para visualización y cálculo de features.

### Límites Dinámicos

Los límites de los parámetros `t0` y `A` se ajustan dinámicamente basándose en los datos observados:
- `t0`: Basado en el rango de fase observado
- `A`: Basado en el rango de flujo observado

Esto mejora la convergencia del MCMC, especialmente para supernovas con características atípicas.

### Reproducibilidad

- Semilla fija configurada en `config.py` (`random_seed = 42`)
- Mismos datos + mismos parámetros = mismos resultados
- Útil para debugging y comparaciones

## Integración con Sistema Existente

Este módulo se integrará con:
- `supernova_classifier.py`: Clasificador de supernovas
- `lightcurve_reader.py`: Lector de curvas de luz (adaptar para formato .dat)
- Modelo de Villar: Para extracción de las 25 features

## Troubleshooting

### Error: "No hay suficientes datos"
- Verifica que el archivo .dat tenga al menos 6 puntos de detección normal (excluyendo upper limits)
- Algunas supernovas pueden tener muy pocos datos
- El código incluye automáticamente los últimos 3 upper limits antes de la primera detección (si están dentro de 20 días) para mejorar el ajuste

### Error: "Initial state has a large condition number"
- El código maneja esto automáticamente con inicialización robusta de walkers
- Si persiste, puede indicar datos problemáticos

### Memoria insuficiente
- El código está optimizado para liberar memoria automáticamente
- Si procesas miles de supernovas, considera procesar en lotes más pequeños
- Usa `--resume` para retomar si se interrumpe

### Gráficos no se guardan
- Verifica permisos de escritura en `outputs/plots/`
- Verifica que el directorio existe

## Referencias

- Modelo de Villar (ALERCE): Modelo para ajuste de curvas de luz de supernovas
- emcee: Biblioteca para MCMC en Python
- ZTF: Zwicky Transient Facility
