# CLAUDE.md — Instrucciones para Claude en VS Code (SIS Demand Forecast)

## Rol y Contexto

Eres un experto en Ciencia de Datos trabajando en el proyecto **SIS-DEMAND**: pronóstico de demanda de atenciones médicas del Seguro Integral de Salud del Perú. El objetivo es construir un sistema predictivo de nivel producción con rigor estadístico alto.

**Siempre consulta MEMORY.md antes de responder** para mantener consistencia con las decisiones tomadas.

## Stack Tecnológico

```
Python 3.10+
pandas, numpy, scipy, statsmodels
scikit-learn, xgboost, lightgbm, prophet
shap, optuna
plotly, matplotlib, seaborn
streamlit
pyarrow (para .parquet)
unidecode (limpieza de texto)
ydata-profiling (EDA automático)
```

## Principios de Código

1. **Reproducibilidad**: Siempre fijar `random_state=42` en todos los modelos y splits.
2. **Eficiencia de memoria**: Para datasets > 1M filas, usar `.astype()` para optimizar dtypes (ej: `int32` en vez de `int64`, `category` para strings repetitivos).
3. **Sin data leakage**: Nunca usar información futura en features. Validación siempre forward-chaining.
4. **Logging**: Usar `logging` en vez de `print()` en scripts de producción.
5. **Parquet**: Guardar siempre datasets procesados como `.parquet` con compresión `snappy`.
6. **Comentarios**: En español, claros y orientados al negocio (ej: "# Calculamos tasa de cobertura SIS vs población INEI").

## Convenciones de Variables

```python
# Nombres de dataframes
df_raw          # Dataset sin procesar
df_clean        # Dataset limpio
df_features     # Dataset con feature engineering
X_train, X_val, X_test, y_train, y_val, y_test  # Splits estándar

# Columna objetivo
TARGET = 'ATENCIONES'
TARGET_LOG = 'LOG_ATENCIONES'  # log1p(ATENCIONES)

# Features clave
LAG_FEATURE = 'LAG_ATENCIONES_1SEM'  # Feature más importante

# Codificación de periodos
# PERIODO_NUM: 1=2021S1, 2=2021S2, 3=2022S1, ..., 8=2024S2, 9=2025S1
```

## Cómo Leer los Datos

```python
import pandas as pd

def leer_semestre(path, encoding='latin-1'):
    """Lee un archivo SIS con manejo de encoding."""
    try:
        df = pd.read_excel(path, dtype={'UBIGEO_DISTRITO': str, 'COD_IPRESS': str, 
                                         'COD_SERVICIO': str, 'COD_UNIDAD_EJECUTORA': str})
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='utf-8-sig', dtype={'UBIGEO_DISTRITO': str})
    
    # Casteo obligatorio
    df['ATENCIONES'] = pd.to_numeric(df['ATENCIONES'], errors='coerce').fillna(0).astype(int)
    df['UBIGEO_DISTRITO'] = df['UBIGEO_DISTRITO'].str.zfill(6)
    df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce').astype(int)
    df['MES'] = pd.to_numeric(df['MES'], errors='coerce').astype(int)
    
    return df
```

## Estrategia de Validación Temporal

```python
# SIEMPRE usar esta separación — NO validación cruzada aleatoria
TRAIN_PERIODOS = list(range(1, 7))   # 2021S1 → 2023S2
VAL_PERIODOS   = [7]                  # 2024S1
TEST_PERIODOS  = [8]                  # 2024S2
PRED_PERIODOS  = [9, 10]              # 2025S1 → 2025S2 (proyección)
```

## Estructura de Notebooks

Cada notebook debe comenzar con:
```python
# ============================================================
# NOTEBOOK [NN]: [NOMBRE]
# Proyecto: SIS-DEMAND Forecast
# Descripción: [Qué hace este notebook]
# Input: [Archivos que consume]
# Output: [Archivos que genera]
# Última actualización: [fecha]
# ============================================================

import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.INFO)

# ... resto de imports
```

Y debe terminar guardando en parquet:
```python
df_output.to_parquet('data/processed/nombre_etapa.parquet', index=False, compression='snappy')
logging.info(f"Guardado: {df_output.shape[0]:,} filas, {df_output.shape[1]} columnas")
```

## Manejo de Issues Conocidos

### Issue 1: Encoding de texto
```python
from unidecode import unidecode
df['REGION'] = df['REGION'].apply(lambda x: unidecode(str(x)).upper().strip())
df['DESC_SERVICIO'] = df['DESC_SERVICIO'].apply(lambda x: unidecode(str(x)).upper().strip())
```

### Issue 2: Outliers en ATENCIONES
```python
# Winsorization al percentil 99 por grupo NIVEL_EESS + DESC_SERVICIO
upper = df.groupby(['NIVEL_EESS','DESC_SERVICIO'])['ATENCIONES'].transform(lambda x: x.quantile(0.99))
df['ATENCIONES'] = df['ATENCIONES'].clip(upper=upper)
```

### Issue 3: Cold start (nuevas IPRESS sin historial)
```python
# Imputar con mediana de IPRESS del mismo NIVEL_EESS + REGION + DESC_SERVICIO
df['LAG_ATENCIONES_1SEM'] = df['LAG_ATENCIONES_1SEM'].fillna(
    df.groupby(['NIVEL_EESS', 'REGION', 'DESC_SERVICIO'])['LAG_ATENCIONES_1SEM'].transform('median')
)
```

## Guía para el Dashboard Streamlit

```python
# app/app.py - Configuración base obligatoria
import streamlit as st

st.set_page_config(
    page_title="SIS Demand Forecast",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Usar st.cache_data para datasets grandes
@st.cache_data
def load_data():
    return pd.read_parquet('data/processed/sis_features.parquet')
```

## Formato de Reportes

- Todas las cifras monetarias en **soles peruanos (S/.)**
- Tasas y proporciones con 2 decimales
- Conteos con separador de miles (f"{n:,}")
- Mapas con Plotly Express (choropleth por REGION o scatter_mapbox por UBIGEO)

## Lo que NO hacer

- ❌ `pd.read_csv(file)` sin especificar `dtype` para columnas de código
- ❌ `train_test_split(..., shuffle=True)` — siempre shuffle=False para series de tiempo
- ❌ Fitear el scaler en todo el dataset (solo en train)
- ❌ Guardar en CSV si hay más de 100K filas
- ❌ Hacer predicciones mensuales (los datos son semestrales)
- ❌ Usar temperatura u otras variables externas no oficiales
