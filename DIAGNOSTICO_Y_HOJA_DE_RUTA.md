# 🏥 SIS-DEMAND: Diagnóstico Técnico y Hoja de Ruta del Proyecto
**Proyecto:** Pronóstico de Demanda de Atenciones Médicas — Seguro Integral de Salud (SIS) - Perú  
**Rol:** Experto en Ciencia de Datos  
**Fecha de análisis:** Abril 2025

---

## 1. CORRECCIONES CRÍTICAS AL ANÁLISIS PRELIMINAR

> ⚠️ Antes de continuar, hay correcciones importantes a lo que ya analizaste:

| Punto Analizado | Tu Versión | Realidad Detectada | Impacto |
|---|---|---|---|
| Periodo de datos | "enero 2025" | **Septiembre 2020** | Bajo (es solo la muestra) |
| Tamaño del dataset | Grande/nacional | **140 filas, 1 región (La Libertad), 1 mes** — Es una muestra mínima | ALTO — necesitas descargar todos los semestres |
| Naturaleza de los datos | "registros de atenciones" | **Ya son datos AGREGADOS** — cuentas pre-sumadas por IPRESS+servicio+demografía | ALTO — afecta la elección de modelos |
| Problema de encoding | "caracteres corruptos (A?O, JUNĂN)" | **En esta muestra NO hay corrupción** — texto limpio | Medio — puede estar en otros archivos |
| ATENCIONES como string | Detectado correctamente | ✅ Confirmado — viene como string, requiere casteo | Bajo |
| Alta cardinalidad COD_IPRESS | Detectado | ✅ Confirmado — 18 IPRESS en muestra; en datos completos serán miles | ALTO |

**Implicación más importante:** Los datos son **conteos agregados**, no registros individuales. Esto cambia el enfoque de modelado: usas modelos de regresión sobre conteos (no clasificación, no clustering de pacientes).

---

## 2. DIAGNÓSTICO DE DATOS — MUESTRA ANALIZADA

### 2.1 Estructura Real del Dataset
```
Filas en muestra:     140
Periodo cubierto:     Septiembre 2020
Región:               La Libertad (solo)
IPRESS únicas:        18 establecimientos
Servicios únicos:     32 tipos de prestación
Variable objetivo:    ATENCIONES (numérico entero, conteo)
```

### 2.2 Hallazgos Exploratorios Validados

**Distribución de ATENCIONES (variable objetivo):**
- Media: 7.8 | Mediana: 2 | Max: 123
- Alta asimetría positiva (skewness) → típico de conteos de salud
- El 75% de registros tiene ≤ 7.25 atenciones → muchos valores pequeños
- Esto sugiere usar **log-transform** o modelos apropiados para conteos (Poisson/NB)

**Hallazgo clave — Concentración de demanda:**
- Hospital de Apoyo Leoncio Prado (NIVEL II): **397 atenciones = 36% del total**
- Solo 2 IPRESS de Nivel II acumulan el **54% de todas las atenciones**
- Nivel I (121 registros) tiene menor volumen pero cubre más establecimientos → efecto de **dispersión geográfica**

**Hallazgo clave — Perfil de servicios:**
- Consulta Externa domina (27%) → demanda transversal a todos los grupos
- Telemonitoreo (11.2%) → relevante post-COVID, servicio emergente
- Salud Reproductiva (9.2%) + Planificación Familiar → sesgo hacia población femenina 18-59 años
- CRED 0-4 años (5.1%) + Estimulación Temprana (6.9%) → foco en primera infancia

**Hallazgo clave — Dependencia demográfica-servicio confirmada:**
- 00-04 años → CRED + Estimulación Temprana (100% concentrado en este grupo, correcto)
- 12-29 años → Planificación Familiar (72 de 101 atenciones)
- 60+ → Solo Consulta Externa (sin servicios preventivos intensivos)
- Este patrón es **predecible y modelable** → alta señal estadística

**Hallazgo clave — Plan de seguro:**
- SIS Gratuito: 97.4% de atenciones → La propuesta de valor de SIS es principalmente población vulnerable
- SIS Para Todos: 2.6% → nicho marginal en esta región rural

---

## 3. OBJETIVO REFINADO DEL PROYECTO (DEFINITIVO)

### 3.1 Problema Central

> **"El SIS asigna recursos (personal médico, farmacia, presupuesto) de manera histórica y uniforme, sin anticipar variaciones en la demanda de atenciones por establecimiento, servicio y perfil poblacional — generando subutilización en algunos IPRESS y saturación en otros."**

### 3.2 Objetivo Analítico (Preciso, Medible)

> **Construir un modelo predictivo que estime el volumen de atenciones (ATENCIONES) por combinación de [IPRESS × SERVICIO × GRUPO_EDAD × SEXO] para el siguiente semestre, con un error MAPE ≤ 20%, utilizando datos históricos del SIS 2021–2024.**

### 3.3 Decisión de Negocio que Habilita

Con el modelo operativo, el SIS puede:
1. **Reasignar horas-médico** desde IPRESS proyectadas con baja demanda hacia las de pico
2. **Dimensionar stock farmacéutico** por servicio y distrito con 6 meses de anticipación
3. **Planificar brigadas preventivas** focalizadas en grupos de edad + región con déficit de cobertura
4. **Priorizar inversión** en IPRESS con tendencia creciente sistemática

### 3.4 Métricas de Éxito del Proyecto

| Métrica | Nivel Mínimo | Nivel Destacado |
|---|---|---|
| MAPE del modelo | < 25% | < 15% |
| R² ajustado | > 0.70 | > 0.85 |
| RMSE vs Baseline (promedio histórico) | Mejora > 15% | Mejora > 30% |
| Feature Importance interpretable | SHAP values básicos | SHAP + partial dependence |
| Propuesta de valor cuantificada | Estimación cualitativa | Ahorro monetario proyectado en S/. |

---

## 4. FUENTES DE DATOS — PLAN DE INGESTA COMPLETA

### 4.1 Dataset Principal (SIS — Datos Abiertos)

**URL Base:** `https://www.datosabiertos.gob.pe/dataset/atenciones-realizadas-los-asegurados-del-seguro-integral-de-salud-sis`

| Archivo | Periodo | Observaciones |
|---|---|---|
| DS_01_020_ATENCIONES_2021_S1 | Ene–Jun 2021 | Primer semestre post-pandemia |
| DS_01_020_ATENCIONES_2021_S2 | Jul–Dic 2021 | Recuperación de demanda |
| DS_01_020_ATENCIONES_2022_S1 | Ene–Jun 2022 | |
| DS_01_020_ATENCIONES_2022_S2 | Jul–Dic 2022 | |
| DS_01_020_ATENCIONES_2023_S1 | Ene–Jun 2023 | |
| DS_01_020_ATENCIONES_2023_S2 | Jul–Dic 2023 | |
| DS_01_020_ATENCIONES_2024_S1 | Ene–Jun 2024 | |
| DS_01_020_ATENCIONES_2024_S2 | Jul–Dic 2024 | |
| DS_01_020_ATENCIONES_2025_S1 | Ene–Jun 2025 | Datos más recientes |

**Tamaño estimado:** Cada semestre puede tener entre 500K – 2M filas (toda la red nacional SIS). Total consolidado: **~5–15 millones de filas**.

### 4.2 Variable Externa Recomendada: Población INEI por Distrito

**Fuente:** INEI — Censo Nacional 2017 y proyecciones 2020–2025  
**URL:** `https://www.inei.gob.pe/estadisticas/indice-tematico/poblacion-y-vivienda/`  
**Por qué SÍ agregarla:**
- Es oficial (INEI), de alta credibilidad
- Permite calcular **tasa de cobertura = ATENCIONES / POBLACIÓN** por grupo de edad y distrito
- Convierte el conteo absoluto en una métrica relativa mucho más útil estratégicamente
- Fácil de hacer join por UBIGEO_DISTRITO (6 dígitos, ya está en el dataset SIS)

**Variables derivables:**
- `POBLACION_DISTRITO_GRUPO_EDAD`: Población por ubigeo y grupo etario
- `TASA_COBERTURA`: ATENCIONES / POBLACION → detecta distritos con baja penetración SIS
- `RATIO_DEMANDA_POTENCIAL`: qué porcentaje de la población no está siendo atendida

### 4.3 Variables Externas NO Recomendadas

| Variable | Motivo de Exclusión |
|---|---|
| Temperatura / clima | Requiere API meteorológica + join geoespacial complejo. Riesgo de introducir ruido. Credibilidad cuestionable sin experto climático. |
| MINSA datos epidemiológicos | Granularidad diferente. Difícil de sincronizar temporalmente. Agrega complejidad sin garantía de mejora. |
| Datos socioeconómicos INEI (pobreza, NBI) | Interesante pero requiere cruce censal complejo. Puede incluirse como versión 2.0 del proyecto. |

> **Decisión de arquitectura:** Usar solo datos SIS + población INEI. Esto mantiene rigor, credibilidad y manejabilidad del proyecto.

---

## 5. ARQUITECTURA DEL PROYECTO — ESTRUCTURA DE CARPETAS

```
sis-demand-forecast/
│
├── data/
│   ├── raw/                    # Archivos descargados sin modificar
│   │   ├── 2021_S1.xlsx
│   │   ├── 2021_S2.xlsx
│   │   └── ... (9 archivos)
│   ├── external/
│   │   └── inei_poblacion_distrito.csv
│   ├── processed/
│   │   ├── sis_consolidado.parquet    # Dataset maestro unificado
│   │   └── sis_features.parquet      # Dataset con features engineered
│   └── outputs/
│       ├── predictions/
│       └── evaluation/
│
├── notebooks/
│   ├── 00_ingesta_consolidacion.ipynb
│   ├── 01_eda_univariado.ipynb
│   ├── 02_eda_bivariado.ipynb
│   ├── 03_eda_multivariado.ipynb
│   ├── 04_limpieza_feature_engineering.ipynb
│   ├── 05_clustering_ipress.ipynb
│   ├── 06_modelado_predictivo.ipynb
│   ├── 07_evaluacion_interpretabilidad.ipynb
│   └── 08_propuesta_valor_cuantificada.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── models.py
│   └── evaluator.py
│
├── app/                        # Streamlit Dashboard
│   ├── app.py
│   ├── pages/
│   │   ├── 01_diagnostico.py
│   │   ├── 02_prediccion.py
│   │   └── 03_clustering.py
│   └── utils/
│
├── models/                     # Modelos serializados
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── reports/
│   └── TAF_SIS_Demand_Forecast.pdf
│
├── MEMORY.md                   # Contexto del proyecto para Claude
├── CLAUDE.md                   # Instrucciones para Claude en VS Code
├── SKILLS.md                   # Pipeline técnico detallado
├── requirements.txt
└── README.md
```

---

## 6. PIPELINE DE CIENCIA DE DATOS — ETAPAS DETALLADAS

### ETAPA 0: Ingesta y Consolidación (`00_ingesta_consolidacion.ipynb`)

**Objetivo:** Descargar y unir los 9 semestres en un único dataset maestro.

```python
# Acciones clave:
# 1. Leer cada archivo con encoding='utf-8' (o 'latin-1' si falla)
# 2. Crear columna SEMESTRE (ej: "2021_S1") y PERIODO_NUM (1–9) para series de tiempo
# 3. Castear ATENCIONES a int, UBIGEO_DISTRITO a str (pad con ceros a 6 dígitos)
# 4. Guardar como .parquet (10x más rápido que CSV para datasets grandes)
# 5. Validar: shape, nulos, rangos de fechas
```

**Issues anticipados:**
- Encoding: probar `latin-1` primero, luego `utf-8-sig` si hay caracteres corruptos
- Cambios de esquema entre semestres (columnas renombradas entre versiones del SIS)
- ATENCIONES como string → `pd.to_numeric(..., errors='coerce')`

### ETAPA 1: EDA Univariado (`01_eda_univariado.ipynb`)

**Para cada variable analizar:**

| Variable | Análisis |
|---|---|
| ATENCIONES | Histograma, boxplot, Q-Q plot, test de normalidad (Shapiro o Kolmogorov-Smirnov), skewness/kurtosis |
| REGION | Frecuencia absoluta y relativa, mapa de calor geográfico |
| GRUPO_EDAD | Pirámide poblacional de demanda |
| DESC_SERVICIO | Pareto 80/20 de servicios |
| NIVEL_EESS | Proporción Nivel I vs II vs III |
| PLAN_DE_SEGURO | Distribución por plan |
| SEXO | Ratio F/M por servicio |

**Output clave:** Tabla de perfilado estadístico completo (pandas-profiling o ydata-profiling)

### ETAPA 2: EDA Bivariado (`02_eda_bivariado.ipynb`)

**Análisis de relaciones clave:**
- ATENCIONES vs GRUPO_EDAD (boxplot por grupo)
- ATENCIONES vs NIVEL_EESS (test de Kruskal-Wallis → diferencias significativas)
- ATENCIONES vs REGION (mapa de calor)
- ATENCIONES vs AÑO_MES (serie de tiempo de tendencias nacionales)
- PLAN_DE_SEGURO vs DESC_SERVICIO (tabla de contingencia + chi-cuadrado)
- Estacionalidad: agrupar por MES y analizar patrones recurrentes

**Hipótesis a probar:**
- H1: Las atenciones en Nivel I son significativamente menores por IPRESS que en Nivel II
- H2: Existe estacionalidad semestral en servicios materno-infantiles
- H3: El crecimiento de telemedicina post-2020 es estadísticamente significativo

### ETAPA 3: EDA Multivariado (`03_eda_multivariado.ipynb`)

- Matriz de correlación (variables numéricas después de encoding)
- Análisis de VIF (Variance Inflation Factor) para detectar multicolinealidad
- PCA para reducción dimensional exploratoria
- Heatmap de REGION × DESC_SERVICIO (demanda por tipo de servicio por región)
- Análisis de interacción GRUPO_EDAD × SEXO × DESC_SERVICIO

### ETAPA 4: Limpieza y Feature Engineering (`04_limpieza_feature_engineering.ipynb`)

**Limpieza:**
```python
# 1. Normalizar strings: strip(), upper(), remover tildes con unidecode
# 2. Castear tipos: ATENCIONES→int, AÑO→int, MES→int, UBIGEO→str.zfill(6)
# 3. Tratar outliers: IQR en ATENCIONES (winsorization al 99th percentile)
# 4. Validar UBIGEO contra tabla oficial INEI (detectar ubigeos fantasma)
```

**Feature Engineering (variables derivadas):**

| Feature Nuevo | Fórmula / Lógica | Justificación |
|---|---|---|
| `PERIODO` | AÑO * 100 + MES | Identificador temporal numérico |
| `SEMESTRE` | 1 si MES ≤ 6 else 2 | Captura estacionalidad semestral |
| `LOG_ATENCIONES` | log1p(ATENCIONES) | Normaliza distribución sesgada |
| `TASA_COBERTURA` | ATENCIONES / POB_DISTRITO_EDAD | Requiere join con INEI — indicador de eficiencia |
| `SERVICIO_CATEGORIA` | Agrupación de 32 → 6 macro-categorías | Reduce cardinalidad: Preventivo, Materno, Pediátrico, Crónico, Urgencias, Telematica |
| `ES_SERVICIO_PREVENTIVO` | 1/0 flag | Servicios de prevención vs curativos |
| `REGION_TIPO` | Costa/Sierra/Selva según región | Proxy geográfico de condiciones sanitarias |
| `LAG_ATENCIONES_1SEM` | ATENCIONES del semestre anterior para misma combinación | Feature más poderoso para time-series |
| `ROLLING_MEAN_2SEM` | Media de últimos 2 semestres | Suaviza estacionalidad |
| `TENDENCIA_LINEAL` | Pendiente de regresión simple sobre serie histórica por IPRESS+SERVICIO | Captura crecimiento/decrecimiento estructural |

**Encoding de variables categóricas:**
- `REGION`: Target Encoding (usando media de ATENCIONES por región)
- `DESC_SERVICIO`: Target Encoding o One-Hot si ≤ 32 categorías
- `GRUPO_EDAD`: Ordinal Encoding (hay un orden natural: 0-4 < 5-11 < ... < 60+)
- `NIVEL_EESS`: Label Encoding (I=1, II=2, III=3)
- `SEXO`: Binary (0/1)
- `PLAN_DE_SEGURO`: One-Hot (pocos valores únicos)

### ETAPA 5: Clustering de IPRESS (`05_clustering_ipress.ipynb`)

**Objetivo:** Segmentar los establecimientos en perfiles de demanda similares para identificar grupos estratégicos.

**Features para clustering:**
- Total atenciones promedio por semestre
- Distribución porcentual de servicios (% materno, % pediátrico, % urgencias, etc.)
- Distribución de grupo de edad dominante
- Nivel EESS
- Tipo geográfico (Costa/Sierra/Selva)

**Algoritmo:** K-Means con k óptimo por método del codo + Silhouette Score  
**Alternativa:** DBSCAN si hay outliers geográficos  
**Resultado esperado:** 4–6 perfiles de IPRESS (ej: "Hospital urbano de alta rotación", "Posta rural pediátrica", "Centro de salud preventivo femenino", etc.)

### ETAPA 6: Modelado Predictivo (`06_modelado_predictivo.ipynb`)

**Formulación del problema:** Regresión sobre `ATENCIONES` (o `LOG_ATENCIONES`)  
**Unidad de predicción:** Combinación IPRESS × SERVICIO × GRUPO_EDAD × SEXO para el siguiente semestre

**Estrategia de validación temporal:**
```
Entrenamiento:    2021 S1 → 2023 S2  (6 semestres)
Validación:       2024 S1             (1 semestre)  
Test final:       2024 S2             (1 semestre)
Predicción:       2025 S1 → S2
```
> ⚠️ NUNCA usar validación cruzada aleatoria (data leakage temporal). Siempre forward-chaining.

**Modelos a entrenar:**

| # | Modelo | Librería | Justificación |
|---|---|---|---|
| M0 | **Baseline: Promedio histórico** | pandas | Benchmark de referencia obligatorio |
| M1 | **Random Forest Regressor** | scikit-learn | Robusto, interpetable con feature importance |
| M2 | **XGBoost Regressor** | xgboost | State-of-the-art para datos tabulares |
| M3 | **LightGBM Regressor** | lightgbm | Más rápido que XGBoost en datasets grandes |
| M4 | **Prophet (por IPRESS+SERVICIO)** | prophet | Captura tendencia + estacionalidad automáticamente |

**Hiperparámetros:** Optimizar M1 y M2 con Optuna (Bayesian Search, 50 trials)

### ETAPA 7: Evaluación e Interpretabilidad (`07_evaluacion_interpretabilidad.ipynb`)

**Métricas:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 evita división por 0
```

**Métricas a reportar:** MAE, RMSE, MAPE, R², Mejora vs Baseline (%)

**Interpretabilidad con SHAP:**
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # Global importance
shap.waterfall_plot(...)                 # Explicación individual
```

**Análisis de errores:** Identificar segmentos donde el modelo falla más (ej: distritos rurales con datos escasos — "cold start problem")

### ETAPA 8: Propuesta de Valor Cuantificada (`08_propuesta_valor_cuantificada.ipynb`)

> Esta es la sección más importante para el TAF. Debe ser cuantitativa y convincente.

**Ejemplo de cuantificación:**
```
Supuesto base:
- Un médico en un centro de salud nivel I cuesta en promedio S/. 8,000/mes al SIS
- Un error de planificación de ±20% en demanda genera sobredotación o infradotación
- Con modelo predictivo: MAPE < 15% vs baseline MAPE ~35%

Cálculo:
- Red nacional SIS: ~7,000 IPRESS
- Si el 10% mejora su asignación de 1 médico/mes: 700 IPRESS × S/. 8,000 × 15% eficiencia = S/. 840,000/mes
- Anualmente: S/. 10.08 millones en optimización de recursos humanos

Adicionalmente:
- Reducción de consultas perdidas por saturación → mejora en indicadores de salud
- Focalización de campañas preventivas → reducción de demanda curativa futura
```

---

## 7. DASHBOARD STREAMLIT — ARQUITECTURA

```python
# app/app.py - Estructura principal

# Página 1: Diagnóstico Nacional
# - Mapa de calor de atenciones por región
# - KPIs: total atenciones, IPRESS activas, cobertura promedio
# - Serie de tiempo 2021-2024

# Página 2: Predicción de Demanda
# - Filtros: Región, IPRESS, Servicio, Semestre a predecir
# - Output: Gráfico de demanda proyectada vs histórica
# - Intervalo de confianza (±1 std del ensemble)

# Página 3: Segmentación de IPRESS
# - Mapa de clusters (scatter con colores)
# - Perfil de cada cluster
# - Recomendación específica por tipo de IPRESS

# Página 4: Alertas Tempranas
# - IPRESS proyectadas con saturación (demanda > capacidad histórica media + 1.5 std)
# - Semáforo: rojo/amarillo/verde por servicio y región
```

---

## 8. EVALUACIÓN vs RÚBRICA — MAPEO

| Criterio Rúbrica | Cómo lo cubrimos | Nivel Esperado |
|---|---|---|
| **Exploración** (5%) | EDA Uni+Bi+Multivariado con tests estadísticos, análisis de causalidad edad→servicio | DESTACADO |
| **Ideación Creativa** (5%) | Propuesta de redistribución dinámica de recursos + alertas tempranas + tasa de cobertura INEI | DESTACADO |
| **Gestión de la Innovación** (5%) | Modelo predictivo + clustering + dashboard Streamlit como producto de datos operativo | DESTACADO |
| **Estructura del TAF** (5%) | 8 secciones completas con cuantificación monetaria de impacto | DESTACADO |

---

## 9. ARCHIVOS DE CONFIGURACIÓN DEL PROYECTO

### 9.1 requirements.txt
```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
prophet>=1.1.4
shap>=0.43.0
optuna>=3.0.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
pyarrow>=13.0.0
openpyxl>=3.1.0
unidecode>=1.3.0
scipy>=1.11.0
statsmodels>=0.14.0
ydata-profiling>=4.5.0
```

---

## 10. RUTA CRÍTICA — TIMELINE RECOMENDADO

```
SEMANA 1: Datos
├── Día 1-2: Descargar los 9 semestres de datos.gob.pe
├── Día 3:   Ejecutar notebook 00 (consolidación + parquet)
└── Día 4-5: Descargar datos INEI + hacer join por UBIGEO

SEMANA 2: EDA y Limpieza
├── Día 1-2: Notebooks 01, 02, 03 (EDA completo)
└── Día 3-5: Notebook 04 (limpieza + feature engineering completo)

SEMANA 3: Modelos
├── Día 1:   Notebook 05 (clustering K-Means)
├── Día 2-3: Notebook 06 (Random Forest + XGBoost + LightGBM)
├── Día 4:   Notebook 07 (evaluación + SHAP)
└── Día 5:   Notebook 08 (cuantificación valor)

SEMANA 4: Entregables
├── Día 1-2: Dashboard Streamlit
├── Día 3-4: TAF documento + video
└── Día 5:   Revisión final y publicación
```

---

## 11. RECOMENDACIONES FINALES COMO EXPERTO

1. **Guarda todo en .parquet**: Para 5-15M de filas, CSV es inaceptablemente lento. Parquet comprimido es 10x más rápido.

2. **El LAG feature es tu arma más poderosa**: `LAG_ATENCIONES_1SEM` (atenciones del semestre anterior para la misma combinación IPRESS+SERVICIO+EDAD+SEXO) explicará el 60-70% de la varianza. Es la clave del modelo.

3. **Clustering antes que predicción**: Los perfiles de IPRESS te permiten contar una historia narrativa poderosa en el video. "Identificamos 5 tipos de centros de salud en el Perú..."

4. **La cuantificación monetaria es lo más importante del TAF**: Busca el presupuesto anual del SIS (MINSA/MEF), calcula qué % de mejora operativa genera tu modelo. Esto transforma tu proyecto de técnico a estratégico.

5. **En el video — Data Storytelling**: Estructura como: problema humano → datos → insight sorprendente → modelo → impacto en vidas/soles → llamado a la acción.

6. **Cuidado con la granularidad temporal**: Los datos son semestrales, no mensuales. No intentes hacer predicciones mensuales — el SIS reporta por semestre. Ajusta tus modelos a esta granularidad.
