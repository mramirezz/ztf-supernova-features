# Extracción de Features de Supernovas ZTF - Literatura

## Resumen del Proyecto

Este módulo está diseñado para extraer features del modelo de Villar a partir de datos de fotometría de supernovas de la literatura ZTF almacenados en formato `.dat`. Utiliza MCMC (Markov Chain Monte Carlo) con `emcee` para ajustar el modelo ALERCE y extraer 25 características de cada curva de luz.

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
├── Photometry_ZTF_ST_Alerce/   # Directorio de datos (debe estar presente)
│   ├── SN Ia/                  # Subcarpetas por tipo de supernova
│   ├── SN II/
│   └── ...
└── outputs/                    # Directorio de salida (generado automáticamente)
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
        ├── {sn_type}_debug.pdf
        ├── {sn_type}_successful.csv
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

## Instalación y Configuración

### Crear Entorno Conda (Recomendado)

```bash
# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate ztf_features/streamlit_env
```

**Nota**: El entorno se llama `ztf_features` y contiene todas las dependencias necesarias (numpy, pandas, matplotlib, scipy, emcee, corner, streamlit).

### Verificar Instalación

```bash
# Verificar que el entorno está activo
conda info --envs

# Verificar paquetes instalados
conda list
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

**Optimización de Memoria:**

El código está optimizado para procesar miles de supernovas sin quedarse sin memoria:

- **Liberación de figuras**: Las figuras de matplotlib se cierran inmediatamente después de guardarlas
- **Eliminación de samples MCMC**: Los samples grandes se eliminan después de generar los plots
- **Recolección de basura**: `gc.collect()` se ejecuta periódicamente (después de cada filtro, cada supernova, y cada 10 supernovas)
- **Sin acumulación**: Los datos se procesan y liberan inmediatamente, no se acumulan en memoria

**Nota**: Cada filtro genera un registro separado en el CSV. Si procesas 3 supernovas con 2 filtros cada una, obtendrás 6 registros en total.

## Modo Debug PDF (--debug-pdf)

**Para generar PDFs de inspección visual con múltiples supernovas:**

El modo debug genera PDFs con fit plots y corner plots para inspección visual rápida de múltiples supernovas. Puede procesar supernovas aleatoriamente o desde un CSV.

**Uso básico (selección aleatoria):**

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

# Reanudar desde checkpoint (añade páginas al PDF existente)
python main.py "SN Ia" 200 --debug-pdf --resume
```

**Uso con CSV (--from-csv):**

Procesa supernovas específicas desde un archivo CSV:

```bash
# Procesar todas las supernovas del CSV
python main.py "SN II" all --debug-pdf --from-csv outputs/debug_pdfs/SN_II_successful.csv

# Sobrescribir PDF y CSV existentes (--overwrite)
python main.py "SN II" all --debug-pdf --from-csv outputs/debug_pdfs/SN_II_successful.csv --overwrite

# Reanudar desde checkpoint (añade páginas al PDF existente)
python main.py "SN II" all --debug-pdf --from-csv outputs/debug_pdfs/SN_II_successful.csv --resume
```

**Formato del CSV:**

El CSV debe tener una columna llamada `supernova_name` con los nombres de las supernovas (sin extensión `.dat`):

```csv
supernova_name
ZTF18aaaibml
ZTF22aaevwec
ZTF23abbtkrv
...
```

**Parámetros del modo CSV:**

- `--from-csv <ruta>`: Archivo CSV con lista de supernovas a procesar
- `--overwrite`: Sobrescribe el PDF y CSV de features existentes (solo tiene efecto con `--from-csv`)
- `--resume`: Reanuda desde checkpoint y añade páginas al PDF existente (no sobrescribe)
- Si usas `--from-csv`, el parámetro `n_supernovas` se ignora (procesa todas las del CSV)
- Si usas `--from-csv`, el filtro de año se ignora (procesa todas las del CSV)

**Características del Modo Debug:**

- **Selección aleatoria**: Selecciona supernovas aleatoriamente (no solo las primeras N)
- **Filtrado por año**: Solo procesa supernovas del año 2022 en adelante (ZTF22, ZTF23, etc.)
- **Validación**: Solo incluye supernovas con al menos 6 detecciones normales (excluyendo upper limits)
- **Continuación automática**: Continúa intentando hasta obtener el número solicitado de supernovas exitosas
- **Checkpointing y guardado incremental**: 
  - Guarda checkpoint después de cada supernova exitosa
  - Guarda cada página al PDF inmediatamente (no espera al final)
  - Si el proceso se interrumpe, puedes reanudar con `--resume` sin perder trabajo
  - Al reanudar, añade nuevas páginas al PDF existente (requiere PyPDF2)
- **Una página por supernova**: Cada página contiene:
  - Si hay 1 filtro: Fit plot (magnitud y flujo) arriba, corner plot abajo
  - Si hay 2 filtros: Fit plot filtro 1, fit plot filtro 2, corner plot (del filtro 1)
- **Rango común de X**: Cuando hay 2 filtros, ambos comparten el mismo rango de MJD para facilitar comparación
- **Eje X en MJD**: Los plots usan fechas originales (MJD) en lugar de fase relativa
- **CSV de supernovas exitosas**: Genera un CSV adicional con la lista de supernovas procesadas exitosamente
- **Corner plot mejorado**: Muestra números pequeños en notación científica cuando es necesario (ej: A = 1.23e-08 en lugar de 0.00)

**Archivos generados:**

**Modo normal (selección aleatoria):**
- **PDF**: `outputs/debug_pdfs/{sn_type}_debug.pdf`
- **CSV**: `outputs/debug_pdfs/{sn_type}_successful.csv` (lista de supernovas exitosas)
- **Checkpoint**: `outputs/checkpoints/debug_checkpoint_{sn_type}.json` (para reanudar)

**Modo CSV (--from-csv):**
- **PDF**: `outputs/debug_pdfs/{sn_type}_debug_from_csv.pdf`
- **CSV de exitosas**: `outputs/debug_pdfs/{sn_type}_successful_from_csv.csv` (lista de supernovas procesadas exitosamente)
- **CSV de features**: `outputs/features/features_{sn_type}_debug_from_csv.csv` (features extraídas, incluyendo `_moc`)
- **CSV de fallidas**: `outputs/debug_pdfs/{sn_type}_failed_from_csv.csv` (lista de supernovas que fallaron con razones)
- **Checkpoint**: `outputs/checkpoints/debug_checkpoint_{sn_type}_from_csv.json` (para reanudar)

**Valores por defecto según tipo:**

- **SN Ia**: 200 supernovas
- **SN Ia-91bg-like, SN Ia-91T-like**: 50 supernovas
- **Otros tipos**: 100 supernovas

**Reanudar desde checkpoint:**

Si el proceso se interrumpe (por ejemplo, por falta de memoria), puedes reanudar usando `--resume`:

```bash
# Reanudar procesamiento (añade páginas al PDF existente)
python main.py "SN Ia" 200 --debug-pdf --resume
```

El sistema:
1. Carga el checkpoint y salta las supernovas ya procesadas
2. Añade nuevas páginas al PDF existente (requiere PyPDF2: `pip install PyPDF2`)
3. Continúa desde donde se quedó

**Notas importantes:**

- **Features en modo CSV**: Cuando usas `--from-csv`, el modo debug SÍ guarda features en un CSV separado (`features_{sn_type}_debug_from_csv.csv`), incluyendo ambas features (sin sufijo y `_moc`)
- **Features en modo normal**: Sin `--from-csv`, el modo debug NO guarda features en el CSV principal (solo genera PDFs)
- **Checkpoint**: El modo debug usa checkpoint para reanudar procesamiento (usa `--resume` para reanudar)
- **Guardado incremental**: El modo debug guarda cada página inmediatamente (no espera al final)
- **--resume vs --overwrite**:
  - `--resume`: Añade nuevas páginas al PDF existente (requiere PyPDF2), no sobrescribe
  - `--overwrite`: Sobrescribe el PDF y CSV existentes (solo funciona con `--from-csv`)
  - Sin ninguno: Si existe un PDF, se sobrescribirá automáticamente
- **Consistencia**: Los plots en el PDF usan el mismo estilo y funciones que el procesamiento normal
- **Método de análisis**: Tanto el modo normal como el modo debug usan las mismas 200 mejores curvas por log-likelihood para calcular median of parameters, median of curves, corner plot y features, garantizando consistencia completa entre ambos modos

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

## Features Extraídas (31 totales)

- **Parámetros principales - Median of Parameters** (6): A, f, t0, t_rise, t_fall, gamma
- **Parámetros - Median of Curves** (6): A_moc, f_moc, t0_moc, t_rise_moc, t_fall_moc, gamma_moc
- **Errores formales** (6): A_err, f_err, t0_err, t_rise_err, t_fall_err, gamma_err
- **Errores MCMC** (6): A_mc_std, f_mc_std, t0_mc_std, t_rise_mc_std, t_fall_mc_std, gamma_mc_std
- **Métricas de ajuste** (3): rms, mad, median_relative_error_pct
- **Características de curva** (2): n_points, time_span
- **Metadatos** (2): sn_name, filter_band

**Nota**: Los parámetros `_moc` (Median of Curves) provienen de la curva más representativa entre las 200 con mejor log-likelihood. Pueden diferir de los parámetros sin sufijo (Median of Parameters) especialmente cuando hay correlaciones fuertes entre parámetros.

## Configuración

### Parámetros MCMC (config.py)

Los parámetros por defecto están en `config.py`:

```python
MCMC_CONFIG = {
    "n_walkers": 100,
    "n_steps": 5000,
    "burn_in": 500,
    "random_seed": 42  # Para reproducibilidad
}
```

### Filtrado de Datos

El código filtra automáticamente los datos para enfocarse en la fase relevante de la supernova:

```python
DATA_FILTER_CONFIG = {
    "max_days_after_peak": 300.0,   # Máximo 300 días después del pico
    "max_days_before_peak": None,   # Sin límite de días antes del pico (incluir todos los datos desde la primera detección)
    "max_days_before_first_obs": 20.0  # Máximo número de días antes de la primera observación para incluir upper limits
}
```

Si después del filtro quedan menos de 6 puntos de detección normal (excluyendo upper limits), la supernova se omite del procesamiento.

### Tratamiento de Upper Limits

#### ¿Qué son los Upper Limits?

Un **upper limit** (límite superior) es una observación donde la supernova **no fue detectada** por el telescopio, pero sabemos que si hubiera sido más brillante que cierto valor, **sí la habríamos detectado**. En otras palabras, es una medida de "no detección" que nos dice que el flujo verdadero de la supernova en ese momento era **menor o igual** a cierto valor límite.

**Ejemplo práctico:**
- Si observamos una supernova en magnitud 22.0 y no la detectamos, pero sabemos que el telescopio puede detectar objetos hasta magnitud 21.5, entonces tenemos un upper limit de magnitud 21.5
- Esto significa que el flujo verdadero era **menor** que el flujo correspondiente a magnitud 21.5
- En términos de flujo: si $F_{21.5}$ es el flujo correspondiente a magnitud 21.5, entonces $F_{\text{verdadero}} \leq F_{21.5}$

#### ¿Por qué son Datos Censurados?

Los upper limits son un tipo especial de dato llamado **dato censurado** porque:

1. **Sabemos que la fuente existe**: Intentamos observarla en un tiempo específico (la supernova existe)
2. **No conocemos su valor exacto**: No la detectamos, por lo que no sabemos su flujo verdadero
3. **Conocemos una restricción**: Sabemos que el flujo verdadero está **por debajo** del límite observado

**Diferencia entre Datos Censurados y Truncados:**
- **Datos censurados** (nuestro caso): La fuente existe pero no conocemos su valor exacto, solo que está por debajo/encima de un límite. Intentamos observarla pero no la detectamos.
- **Datos truncados**: No sabemos si la fuente existe fuera del rango observable (no se intentó observar fuera del rango, así que no sabemos si existe o no).

#### ¿Por qué Requieren Tratamiento Especial en MCMC?

Los upper limits requieren un tratamiento especial en el ajuste MCMC porque **no podemos usar el mismo likelihood que para detecciones normales**:

- **Detecciones normales**: Tenemos un valor medido con su error, y podemos calcular $\chi^2 = \left(\frac{F_{\text{obs}} - F_{\text{modelo}}}{\sigma}\right)^2$
- **Upper limits**: No tenemos un valor medido, solo sabemos que $F_{\text{verdadero}} \leq F_{\text{ul}}$. No podemos calcular un $\chi^2$ estándar.

Por lo tanto, necesitamos un método estadísticamente riguroso para incorporar esta información en el ajuste MCMC.

#### Inclusión de Upper Limits en el Ajuste

El código incluye automáticamente los **últimos 3 upper limits** que ocurren **antes de la primera detección normal**, siempre que estén dentro del rango configurado en `DATA_FILTER_CONFIG["max_days_before_first_obs"]` (por defecto: 20 días antes de la primera observación).

**¿Por qué solo los primeros?**
- Estos upper limits proporcionan información valiosa sobre el flujo **antes de la explosión**, ayudando a constreñir mejor el modelo en la fase temprana de la curva de luz
- Los upper limits después de la primera detección no se incluyen en el ajuste MCMC, ya que el modelo ya está bien constreñido por las detecciones normales
- Incluir todos los upper limits podría sobre-constreñir el modelo y hacer el ajuste menos robusto

#### Tratamiento en el Ajuste MCMC: Método CDF para Datos Censurados

El MCMC evalúa qué tan bien se ajusta el modelo mediante el **log-posterior**, que combina el **prior** y el **likelihood**. El likelihood a su vez combina contribuciones de detecciones normales y upper limits.

**Estructura General del Ajuste:**

El log-posterior se calcula como:

$$\log P(\theta | \text{datos}) = \log P(\theta) + \log L(\theta | \text{datos})$$

donde:
- $\theta = (A, f, t_0, t_{\text{rise}}, t_{\text{fall}}, \gamma)$ son los parámetros del modelo ALERCE
- $\log P(\theta)$ es el log-prior (con restricciones de upper limits)
- $\log L(\theta | \text{datos})$ es el log-likelihood (con método CDF para upper limits)

**Likelihood para Detecciones Normales:**

Cada detección normal contribuye con chi-cuadrado estándar. Para una detección en tiempo $t_i$ con flujo observado $F_{\text{obs},i}$ y error $\sigma_i$:

$$\chi^2_i = \left(\frac{F_{\text{obs},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)^2$$

El log-likelihood para todas las detecciones normales es:

$$\log L_{\text{normal}} = -\frac{1}{2} \sum_{i \in \text{normales}} \chi^2_i = -\frac{1}{2} \sum_{i \in \text{normales}} \left(\frac{F_{\text{obs},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)^2$$

**Likelihood para Upper Limits: Método CDF (Función de Distribución Acumulada)**

El código implementa el método estadísticamente riguroso basado en la **función de distribución acumulada (CDF)** para datos censurados, siguiendo la metodología descrita en Ivezić et al. (2014), Capítulo 4.2.7.

**Fundamento Teórico:**

Para un upper limit, no sabemos el flujo verdadero $F_{\text{verdadero}}(t_i)$, solo sabemos que $F_{\text{verdadero}}(t_i) \leq F_{\text{ul},i}$. El likelihood correcto es la **probabilidad acumulada** de que el flujo verdadero esté por debajo del límite, dado el modelo:

$$P(F_{\text{verdadero}}(t_i) \leq F_{\text{ul},i} | \text{modelo}) = \Phi\left(\frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)$$

donde:
- $F_{\text{modelo}}(t_i)$ es el flujo predicho por el modelo en tiempo $t_i$
- $F_{\text{ul},i}$ es el valor del upper limit en tiempo $t_i$
- $\sigma_i$ es la incertidumbre asociada al upper limit
- $\Phi(z)$ es la función de distribución acumulada (CDF) de la distribución normal estándar

**Interpretación:**

- Si el modelo predice un flujo **muy por debajo** del límite: $F_{\text{modelo}} \ll F_{\text{ul}}$, entonces $\Phi(z) \to 1$ y $\log L_{\text{ul}} \to 0$ (alta probabilidad, no penaliza)
- Si el modelo predice un flujo **cerca** del límite: $F_{\text{modelo}} \approx F_{\text{ul}}$, entonces $\Phi(z) \approx 0.5$ y $\log L_{\text{ul}} \approx -0.69$ (probabilidad moderada)
- Si el modelo predice un flujo que **excede** el límite: $F_{\text{modelo}} > F_{\text{ul}}$, entonces $\Phi(z) \to 0$ y $\log L_{\text{ul}} \to -\infty$ (probabilidad muy baja, fuerte penalización)

**Cálculo del Log-Likelihood para Upper Limits:**

Para cada upper limit individual en tiempo $t_i$:

$$z_i = \frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}$$

$$\log L_{\text{ul},i} = \ln\left[\Phi(z_i)\right] = \ln\left[\Phi\left(\frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)\right]$$

El log-likelihood total para todos los upper limits es:

$$\log L_{\text{ul}} = \sum_{i \in \text{upper limits}} \log L_{\text{ul},i} = \sum_{i \in \text{upper limits}} \ln\left[\Phi\left(\frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)\right]$$

**Estimación de $\sigma_i$:**

Si no tenemos una incertidumbre específica para cada upper limit (que es el caso típico), se estima como:

$$\sigma_i = \begin{cases}
\sigma_{\text{err},i} & \text{si está disponible y es finito} \\
0.05 \times F_{\text{ul},i} & \text{en caso contrario (5\% del límite)}
\end{cases}$$

El factor del 5% (más estricto que el 10% original) hace que la penalización sea más fuerte cuando el modelo excede el límite, ya que hace que $z_i$ sea más negativo.

**Protecciones Numéricas:**

Para evitar problemas numéricos:
- $z_i$ se limita al rango $[-20, 10]$ (en lugar de $[-10, 10]$) para permitir penalizaciones más extremas
- $\Phi(z_i)$ tiene un mínimo de $10^{-15}$ (equivalente a ~8$\sigma$) para evitar $\log(0) = -\infty$

**Log-Likelihood Total:**

El log-likelihood total combina las contribuciones de detecciones normales y upper limits:

$$\log L(\theta | \text{datos}) = \log L_{\text{normal}} + \log L_{\text{ul}}$$

$$\log L(\theta | \text{datos}) = -\frac{1}{2} \sum_{i \in \text{normales}} \left(\frac{F_{\text{obs},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)^2 + \sum_{i \in \text{upper limits}} \ln\left[\Phi\left(\frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)\right]$$

**Referencia Bibliográfica:**

- **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy: A Practical Python Guide for the Analysis of Survey Data*. Princeton University Press.
  - Capítulo 4.2.7: "The MLE in the Case of Truncated and Censored Data" (páginas 130-131)
  - Capítulo 8.1: "Formulation of the Regression Problem" (páginas 322-324)

**Ejemplo Numérico del Método CDF:**

Consideremos un upper limit típico con $F_{\text{ul}} = 1.0 \times 10^{-8}$ y $\sigma = 0.05 \times F_{\text{ul}} = 5.0 \times 10^{-10}$:

| $F_{\text{modelo}}$ | $z = \frac{F_{\text{ul}} - F_{\text{modelo}}}{\sigma}$ | $\Phi(z)$ | $\log L_{\text{ul}}$ | Interpretación |
|---------------------|--------------------------------------------------------|-----------|----------------------|----------------|
| $0.5 \times 10^{-8}$ | $10.0$ | $\approx 1.0$ | $\approx 0$ | No penaliza (modelo muy por debajo) |
| $0.9 \times 10^{-8}$ | $2.0$ | $\approx 0.977$ | $\approx -0.023$ | Penalización muy leve |
| $1.0 \times 10^{-8}$ | $0.0$ | $0.5$ | $-0.693$ | Probabilidad moderada |
| $1.1 \times 10^{-8}$ | $-2.0$ | $\approx 0.023$ | $\approx -3.78$ | Penalización fuerte (excede 10%) |
| $2.0 \times 10^{-8}$ | $-20.0$ | $\approx 10^{-15}$ | $\approx -34.5$ | Penalización extrema (excede 100%) |

**Prior Restrictivo para Upper Limits:**

Además de la penalización en el likelihood, el código implementa un **prior restrictivo** que penaliza fuertemente parámetros que hacen que el modelo exceda upper limits. Esto es más estricto que solo la penalización en el likelihood y ayuda a "cortar" el espacio de parámetros, como se describe en Ivezić et al. (2014), Capítulo 8.

El prior se define como:

$$P(\theta) = \begin{cases} 
0 & \text{si } \theta \text{ está dentro de bounds y } \forall i: F_{\text{modelo}}(t_i) \leq F_{\text{ul},i} \\
-\infty & \text{si } \theta \text{ está fuera de bounds o modelo inválido} \\
-\alpha \sum_{i: F_{\text{modelo}}(t_i) > F_{\text{ul},i}} \left(\frac{F_{\text{modelo}}(t_i) - F_{\text{ul},i}}{F_{\text{ul},i}}\right)^2 & \text{si } \exists i: F_{\text{modelo}}(t_i) > F_{\text{ul},i}
\end{cases}$$

donde:
- $\theta = (A, f, t_0, t_{\text{rise}}, t_{\text{fall}}, \gamma)$ son los parámetros del modelo
- $F_{\text{modelo}}(t_i)$ es el flujo predicho por el modelo en tiempo $t_i$ donde hay un upper limit
- $F_{\text{ul},i}$ es el valor del upper limit en tiempo $t_i$
- $\alpha = 10^{10}$ es un factor de penalización muy grande (pero finito para evitar problemas numéricos)

En términos del log-prior:

$$\log P(\theta) = \begin{cases} 
0 & \text{si modelo respeta todos los upper limits} \\
-\alpha \sum_{i: \text{exceso}_i > 0} \left(\frac{\text{exceso}_i}{F_{\text{ul},i}}\right)^2 & \text{si modelo excede algún upper limit}
\end{cases}$$

donde $\text{exceso}_i = F_{\text{modelo}}(t_i) - F_{\text{ul},i}$ es el exceso del modelo sobre el upper limit.

**Ventajas del Prior Restrictivo:**

1. **Más estricto que solo el likelihood**: El prior rechaza explícitamente (con penalización muy fuerte) parámetros que exceden upper limits, no solo los penaliza suavemente
2. **Independiente de la escala**: La penalización es relativa al exceso normalizado $\left(\frac{\text{exceso}}{F_{\text{ul}}}\right)^2$, por lo que funciona igual para flujos de cualquier escala (típicamente ~$10^{-8}$ para supernovas)
3. **Robusto numéricamente**: Usa una penalización finita ($-10^{10}$) en lugar de $-\infty$ para evitar problemas numéricos en `emcee` cuando muchos walkers tienen `log_prob = -∞`
4. **Corta el espacio de parámetros**: Similar a los "half planes" descritos en Ivezić et al. (2014), el prior efectivamente corta el espacio de parámetros, restringiendo la exploración MCMC a regiones válidas

**Ejemplo Numérico del Prior:**

Para un upper limit con $F_{\text{ul}} = 1.0 \times 10^{-8}$:

| $F_{\text{modelo}}$ | Exceso Relativo | Penalización del Prior | Interpretación |
|---------------------|-----------------|------------------------|----------------|
| $0.5 \times 10^{-8}$ | 0 | 0 | No penaliza (modelo por debajo) |
| $1.1 \times 10^{-8}$ | 0.1 | $-10^{10} \times 0.01 = -10^8$ | Penalización muy fuerte (excede 10%) |
| $2.0 \times 10^{-8}$ | 1.0 | $-10^{10} \times 1.0 = -10^{10}$ | Penalización extrema (excede 100%) |

**¿Por qué Penalización Cuadrática del Exceso Relativo?**

La penalización cuadrática del exceso relativo $\left(\frac{\text{exceso}}{F_{\text{ul}}}\right)^2$ tiene propiedades importantes que la hacen ideal para este propósito:

1. **Normalización Relativa**: Al dividir el exceso absoluto por el valor del upper limit, obtenemos un exceso **relativo** (adimensional). Esto significa que un exceso del 10% siempre da la misma penalización relativa, independientemente de si el upper limit es $10^{-8}$ o $10^{-6}$. Esto hace que el método sea **independiente de la escala** de los flujos.

2. **Penalización Cuadrática**: El cuadrado del exceso relativo hace que la penalización crezca **rápidamente** con el exceso. Por ejemplo:
   - Exceso del 10%: $(0.1)^2 = 0.01$ → penalización proporcional a $0.01$
   - Exceso del 20%: $(0.2)^2 = 0.04$ → penalización proporcional a $0.04$ (4 veces mayor)
   - Exceso del 50%: $(0.5)^2 = 0.25$ → penalización proporcional a $0.25$ (25 veces mayor)
   
   Esto significa que **pequeños excesos son tolerables, pero excesos grandes son severamente penalizados**, lo cual es físicamente razonable: un modelo que excede un upper limit en un 5% es menos problemático que uno que lo excede en un 50%.

3. **Combinación con Factor de Escala**: El factor $\alpha = 10^{10}$ multiplica la penalización cuadrática, asegurando que **cualquier exceso, por pequeño que sea, resulte en una penalización extremadamente fuerte**. Esto hace que el prior efectivamente "corte" el espacio de parámetros, rechazando prácticamente cualquier conjunto de parámetros que cause que el modelo exceda un upper limit.

**Ejemplo Detallado del Cálculo:**

Consideremos un caso donde el modelo excede un upper limit:

- Upper limit: $F_{\text{ul}} = 1.0 \times 10^{-8}$
- Flujo del modelo: $F_{\text{modelo}} = 1.2 \times 10^{-8}$ (excede 20%)

El cálculo de la penalización es:

1. **Exceso absoluto**: $\text{exceso} = F_{\text{modelo}} - F_{\text{ul}} = 1.2 \times 10^{-8} - 1.0 \times 10^{-8} = 0.2 \times 10^{-8}$

2. **Exceso relativo**: $\frac{\text{exceso}}{F_{\text{ul}}} = \frac{0.2 \times 10^{-8}}{1.0 \times 10^{-8}} = 0.2$ (20%)

3. **Exceso relativo al cuadrado**: $\left(\frac{\text{exceso}}{F_{\text{ul}}}\right)^2 = (0.2)^2 = 0.04$

4. **Penalización final**: $\text{penalización} = -\alpha \times 0.04 = -10^{10} \times 0.04 = -4 \times 10^8$

Esta penalización de $-4 \times 10^8$ es **muchísimo más fuerte** que cualquier contribución del likelihood (que típicamente está en el rango de $-100$ a $-10$), por lo que el MCMC prácticamente rechazará este conjunto de parámetros.

**Comparación con el Likelihood CDF:**

Es importante notar que el prior cuadrático y el likelihood CDF trabajan de manera complementaria:

- **Likelihood CDF**: Proporciona una penalización **suave y continua** que incorpora la incertidumbre estadística. Es estadísticamente riguroso y da información incluso cuando el modelo está cerca del límite pero no lo excede.

- **Prior Cuadrático**: Proporciona una penalización **abrupta y extrema** que actúa como un "muro" en el espacio de parámetros. Solo se activa cuando el modelo excede el límite, pero cuando lo hace, la penalización es tan fuerte que prácticamente rechaza esos parámetros.

La combinación de ambos asegura que:
1. El modelo sea guiado suavemente por el likelihood cuando está cerca del límite
2. El modelo sea rechazado fuertemente por el prior cuando excede el límite
3. El espacio de parámetros válido esté bien definido y restringido

#### Posterior Completo

El **log-posterior** combina el prior y el likelihood según el teorema de Bayes:

$$\log P(\theta | \text{datos}) = \log P(\theta) + \log L(\theta | \text{datos}) + \text{constante}$$

donde la constante (evidencia) no depende de $\theta$ y puede ignorarse en el MCMC.

**Desarrollo Completo:**

$$\log P(\theta | \text{datos}) = \log P(\theta) + \log L_{\text{normal}} + \log L_{\text{ul}}$$

Sustituyendo las expresiones:

$$\log P(\theta | \text{datos}) = \log P(\theta) - \frac{1}{2} \sum_{i \in \text{normales}} \left(\frac{F_{\text{obs},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)^2 + \sum_{i \in \text{upper limits}} \ln\left[\Phi\left(\frac{F_{\text{ul},i} - F_{\text{modelo}}(t_i)}{\sigma_i}\right)\right]$$

donde:
- $\log P(\theta)$ es el log-prior (con restricción de upper limits, ver sección siguiente)
- El segundo término es el log-likelihood de detecciones normales (chi-cuadrado)
- El tercer término es el log-likelihood de upper limits (método CDF)

**Implicaciones para el MCMC:**

El MCMC busca maximizar el log-posterior, por lo que:
- El prior restrictivo evita que el MCMC explore regiones del espacio de parámetros donde el modelo excede upper limits
- La combinación de prior restrictivo + likelihood CDF asegura que el modelo respete los upper limits tanto en la exploración como en las estadísticas finales
- Si el modelo es consistente con todos los upper limits, estos no afectan el ajuste (contribución = 0 en ambos, prior y likelihood)
- El filtrado adicional de samples inválidos (después del MCMC) asegura que la mediana/promedio de parámetros respete los upper limits, incluso si algunos samples individuales los exceden debido a correlaciones entre parámetros

**Visualización:**

En los gráficos, los upper limits se muestran como triángulos invertidos:
- **Triángulos verdes**: Upper limits incluidos en el ajuste MCMC (los 3 últimos antes de la primera detección)
- **Triángulos naranjas**: Otros upper limits mostrados para contexto visual pero no usados en el ajuste

### Configuración de Priors (Bounds de Parámetros)

El MCMC utiliza **priors uniformes** (flat priors) dentro de los bounds definidos. Los bounds determinan el espacio de parámetros que el MCMC puede explorar:

```python
MODEL_CONFIG = {
    "param_names": ["A", "f", "t0", "t_rise", "t_fall", "gamma"],
    "bounds": {
        "A": (1e-10, 1e-5),      # Amplitud típica de supernovas
        "f": (0.0, 1.0),          # Fracción de plateau (0 a 1)
        "t0": (-200.0, 50.0),     # Tiempo de referencia: -200 a +50 días
        "t_rise": (1.0, 100.0),   # Tiempo de subida: 1-100 días
        "t_fall": (1.0, 200.0),   # Tiempo de caída: 1-200 días
        "gamma": (0.5, 100.0)     # Gamma: 0.5-100 días
    }
}
```

**Significado de cada bound:**

| Parámetro | Rango | Justificación Física |
|-----------|-------|----------------------|
| $A$ | $[10^{-10}, 10^{-5}]$ | Amplitud típica del flujo de supernovas en unidades normalizadas |
| $f$ | $[0, 1]$ | Fracción del plateau, por definición entre 0 y 1 |
| $t_0$ | $[-200, 50]$ días | Tiempo de explosión relativo, desde -200 a +50 días de la primera detección |
| $t_{\text{rise}}$ | $[1, 100]$ días | Tiempo de subida: desde explosiones rápidas (1 día) hasta lentas (100 días) |
| $t_{\text{fall}}$ | $[1, 200]$ días | Tiempo de caída: desde decaimiento rápido hasta lento |
| $\gamma$ | $[0.5, 100]$ días | Escala temporal del plateau |

**Importancia de los bounds:**
- **Bounds muy amplios** → Mayor incertidumbre en el fit (bandas de confianza anchas)
- **Bounds muy estrechos** → Menor incertidumbre pero riesgo de sesgo si los bounds son incorrectos
- Los valores actuales son un balance basado en el comportamiento físico típico de supernovas tipo II

### Bandas de Confianza (Intervalos de Credibilidad)

Los gráficos muestran **bandas de confianza** que representan la incertidumbre del modelo MCMC. Estas se calculan a partir de los **percentiles de las curvas** generadas por las mejores samples del MCMC.

**Cálculo de las Bandas:**

1. Se seleccionan las **200 curvas con mejor log-likelihood** (de ~2000 candidatas evaluadas)
2. Para cada sample $\theta_k$ de las 200 mejores, se evalúa el modelo completo: $F_k(t) = F_{\text{modelo}}(t; \theta_k)$
3. Para cada tiempo $t$, se calculan los percentiles de los flujos $\{F_1(t), F_2(t), ..., F_{200}(t)\}$

**Nota**: Al usar solo las 200 mejores curvas (en lugar de todas o una muestra aleatoria), las bandas de confianza representan la incertidumbre entre las soluciones de alta calidad, lo que proporciona una estimación más realista de la incertidumbre del modelo.

**Definición Matemática:**

Para un tiempo $t$ dado, si tenemos $N$ curvas de flujo $\{F_1(t), F_2(t), ..., F_N(t)\}$:

- **Banda 68% CI (1σ)**: $[\text{percentil}_{16}(F), \text{percentil}_{84}(F)]$
- **Banda 95% CI (2σ)**: $[\text{percentil}_{2.5}(F), \text{percentil}_{97.5}(F)]$

**¿Por qué estos percentiles?**

Los intervalos del 68% y 95% corresponden a ±1σ y ±2σ de una distribución gaussiana:

| Intervalo | Percentiles | Equivalente Gaussiano | Probabilidad |
|-----------|-------------|----------------------|--------------|
| **68% CI** | 16% - 84% | ±1σ | 68.27% |
| **95% CI** | 2.5% - 97.5% | ±2σ | 95.45% |

**Interpretación Bayesiana:**

En el contexto de MCMC bayesiano, estos son **intervalos de credibilidad** (credible intervals), no intervalos de confianza frecuentistas. La interpretación es directa:

> "Hay un 68% de probabilidad de que la curva verdadera esté dentro de la banda roja oscura"

Esto es más intuitivo que la interpretación frecuentista tradicional.

**¿Por qué usamos 68% como estándar?**

El intervalo del 68% (±1σ) es la convención estándar en física y astronomía porque:
1. Es suficientemente estrecho para ser informativo
2. No es tan estrecho que excluya demasiada incertidumbre
3. Corresponde a la "barra de error" típica reportada en papers científicos

Cuando se reporta un parámetro como $A = 1.15 \times 10^{-7} \pm 0.3 \times 10^{-7}$, el error corresponde al intervalo del 68%.

**Visualización en los Gráficos:**

- **Línea verde sólida**: Curva más central del ensemble ("Median of Curves") - calculada de las 200 mejores curvas
- **Línea azul punteada**: Curva calculada con la mediana de cada parámetro ("Median of Parameters") - calculada de las 200 mejores curvas
- **Líneas rojas individuales**: Las 200 curvas con mejor log-likelihood ploteadas individualmente (con transparencia)
- **Banda roja oscura (68% CI)**: Región donde está el 68% de las 200 mejores curvas
- **Banda roja clara (95% CI)**: Región donde está el 95% de las 200 mejores curvas

### Dos Métodos para la Curva Central: Median of Parameters vs Median of Curves

Los gráficos muestran **dos líneas** que representan diferentes formas de calcular la "curva central" del ajuste MCMC:

**1. Median of Parameters (Línea Azul Punteada)**

Se calcula seleccionando primero las **200 curvas con mejor log-likelihood** (de ~2000 candidatas evaluadas), y luego tomando la **mediana de cada parámetro independientemente** de esas 200 mejores:

$$\theta_{\text{mediana}} = (\text{median}(A), \text{median}(f), \text{median}(t_0), \text{median}(t_{\text{rise}}), \text{median}(t_{\text{fall}}), \text{median}(\gamma))$$

Luego se evalúa el modelo con estos parámetros: $F_{\text{azul}}(t) = F_{\text{modelo}}(t; \theta_{\text{mediana}})$

- **Ventajas**: 
  - Es el método matemáticamente estándar y es lo que se reporta en publicaciones científicas
  - Los valores de los parámetros coinciden exactamente con los mostrados en el corner plot
  - Al usar solo las 200 mejores curvas, se enfoca en soluciones que ajustan bien los datos
- **Desventajas**: Cuando hay **alta correlación entre parámetros** o **alta incertidumbre** en las distribuciones, la combinación de medianas independientes puede no representar una curva "típica" del ensemble de curvas MCMC. En casos extremos, la mediana de parámetros puede producir una curva que ningún sample individual del MCMC produciría.

**2. Median of Curves / Most Central Curve (Línea Verde Sólida)**

Se calcula seleccionando las **mejores curvas según su log-likelihood** y luego encontrando la más representativa:

1. Se evalúan ~2000 samples candidatos del MCMC
2. Para cada candidato $\theta_k$, se calcula su **log-likelihood gaussiano**:
   $$\log L_k = -\frac{1}{2} \sum_i \left(\frac{F_{\text{obs},i} - F_{\text{modelo}}(t_i; \theta_k)}{\sigma_i}\right)^2$$
3. Se seleccionan las **200 curvas con mayor log-likelihood** (mejor ajuste a los datos)
4. Para estas 200 mejores curvas, se calcula la **mediana punto a punto**: $F_{\text{p50}}(t) = \text{percentil}_{50}(\{F_1(t), ..., F_{200}(t)\})$
5. Se identifica cuál de las 200 curvas reales está **más cerca** de la mediana:
   $$k^* = \arg\min_k \sum_t (F_k(t) - F_{\text{p50}}(t))^2$$
6. La curva verde es $F_{\text{verde}}(t) = F_{k^*}(t)$

- **Ventajas**: 
  - La curva resultante es **físicamente realizable** (corresponde a un conjunto real de parámetros del MCMC)
  - Al usar log-likelihood, solo considera curvas que **ajustan bien los datos observados**
  - Evita incluir curvas de regiones del espacio de parámetros que no ajustan bien
  - Es más representativa del "centro" del ensemble de **buenos ajustes**
- **Desventajas**: No corresponde directamente a un conjunto único de parámetros "medianos"

**Parámetros de la Curva Central (MoC):**

Los 6 parámetros de la curva central se guardan en el CSV con sufijo `_moc` (e.g., `A_moc`, `f_moc`, `t0_moc`, etc.), permitiendo comparar con los parámetros de la mediana tradicional.

**Consistencia en el Análisis:**

**Importante**: Las mismas **200 mejores curvas** (seleccionadas por log-likelihood) se usan para calcular:
- **Median of Parameters** (línea azul, features sin sufijo, corner plot)
- **Median of Curves** (línea verde, features con sufijo `_moc`)
- **Corner plot** (histogramas de las 200 mejores curvas)
- **Líneas rojas individuales** (las 200 curvas ploteadas)

Esto garantiza consistencia entre todas las visualizaciones y features extraídas. Ambos métodos (Median of Parameters y Median of Curves) se calculan a partir del mismo conjunto de 200 curvas de alta calidad, lo que permite una comparación justa entre ambos enfoques.

**¿Cuándo Difieren Significativamente?**

Las dos curvas son muy similares cuando:
- Las distribuciones de parámetros son aproximadamente gaussianas
- No hay correlaciones fuertes entre parámetros
- La incertidumbre es baja

Las dos curvas pueden diferir significativamente cuando:
- Hay **correlaciones fuertes** entre parámetros (especialmente $t_0$, $t_{\text{rise}}$, $\gamma$)
- Las distribuciones son **asimétricas o multimodales**
- La **incertidumbre es alta** (bandas de confianza anchas)

**Recomendación:**

Para publicaciones científicas, se recomienda reportar los **parámetros de la mediana** (línea azul) ya que son interpretables y corresponden a lo mostrado en el corner plot. Sin embargo, la línea verde proporciona una mejor representación visual de la "curva típica" cuando hay alta incertidumbre o correlaciones.

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

- **Semilla aleatoria**: Configurada en `config.py` (`MCMC_CONFIG["random_seed"] = 42`)
- **Resultados consistentes**: Con la misma semilla, los resultados MCMC son reproducibles
- **Mismo ajuste**: Si ejecutas dos veces con los mismos datos y parámetros, obtienes el mismo resultado
- **Cambiar semilla**: Puedes especificar una semilla diferente desde la línea de comandos usando `--seed <número>` o `--random-seed <número>`
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
