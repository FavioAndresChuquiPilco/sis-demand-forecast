# MEMORY.md — SIS Demand Forecast Project
> Este archivo es el contexto persistente del proyecto. Actualízalo al completar cada etapa.

## Contexto del Proyecto

**Nombre:** SIS-DEMAND: Pronóstico de Demanda de Atenciones Médicas — SIS Perú  
**Objetivo:** Predecir el volumen de atenciones médicas (ATENCIONES) por combinación [IPRESS × SERVICIO × GRUPO_EDAD × SEXO] para el siguiente semestre, con MAPE ≤ 20%, usando datos históricos SIS 2021–2024.  
**Empresa:** Seguro Integral de Salud (SIS) — Perú  
**Fuente de datos:** Plataforma de Datos Abiertos del Gobierno del Perú  
**URL datos:** https://www.datosabiertos.gob.pe/dataset/atenciones-realizadas-los-asegurados-del-seguro-integral-de-salud-sis

## Dataset Principal — Datos Reales Confirmados (Notebook 00)

- **Archivo consolidado:** `data/processed/sis_consolidado.parquet`
- **Filas totales:** 41,997,568 (≈42M registros)
- **Columnas:** 20
- **Total atenciones acumuladas 2021–2025S1:** 353,696,545 (353.7M)
- **IPRESS únicas:** 8,725
- **Regiones:** 26
- **Servicios distintos:** 65 (≠ 32 mencionado antes — dato real)
- **Periodo disponible:** 9 semestres (PERIODO_NUM 1→9, todos completos)

### Columnas reales del dataset (nombres post-normalización unidecode)
```
ANO, MES, REGION, PROVINCIA, UBIGEO_DISTRITO, DISTRITO,
COD_UNIDAD_EJECUTORA, DESC_UNIDAD_EJECUTORA, COD_IPRESS, IPRESS,
NIVEL_EESS, PLAN_SEGURO,  ← OJO: es PLAN_SEGURO (no PLAN_DE_SEGURO)
COD_SERVICIO, DESC_SERVICIO, SEXO, GRUPO_EDAD, ATENCIONES,
SEMESTRE_LABEL, PERIODO_NUM, SEMESTRE
```

### Filas por semestre
| SEMESTRE_LABEL | PERIODO_NUM | Filas |
|---|---|---|
| 2021_S1 | 1 | 3,338,348 |
| 2021_S2 | 2 | 3,832,485 |
| 2022_S1 | 3 | 4,247,280 |
| 2022_S2 | 4 | 4,418,825 |
| 2023_S1 | 5 | 5,024,801 |
| 2023_S2 | 6 | 5,001,987 |
| 2024_S1 | 7 | 5,352,228 |
| 2024_S2 | 8 | 5,205,985 |
| 2025_S1 | 9 | 5,575,629 |

### Estadísticas de ATENCIONES (datos reales)
- **min:** 1 (no hay registros con 0 atenciones — ya filtrados en origen)
- **media:** 8.42 | **mediana:** 3 | **p90:** 17 | **p95:** 30 | **p99:** 85 | **max:** 9,923
- Alta asimetría positiva confirmada → usar log1p en modelado
- **Top regiones:** Lima Metropolitana (63.3M), Cajamarca (27.2M), Áncash (19.2M), Cusco (18.7M), La Libertad (18.2M)

## Decisiones Técnicas Tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Formato de almacenamiento | .parquet (snappy) | 10x más rápido que CSV; 42M filas × 20 cols |
| Fuente de datos | CSV UTF-8 (no XLSX) | Archivos reales de datos.gob.pe en CSV |
| Encoding real | UTF-8 | Confirmado en notebook 00 |
| Validación temporal | Forward-chaining (train P1–P6, val P7, test P8) | Evitar data leakage temporal |
| Transform target | log1p(ATENCIONES) | Mediana=3, max=9923 — alta asimetría confirmada |
| Encoding categórico | Target encoding (REGION, DESC_SERVICIO), Ordinal (GRUPO_EDAD), Binary (SEXO) | Balance bias-variance |
| Encoding UBIGEO | str.zfill(6) — tratar como categoría, NO numérico | Evitar cálculos sin sentido |
| Variable externa | Solo INEI Población por distrito | Credibilidad oficial, fácil join por UBIGEO |
| Modelos | RF + XGBoost + LightGBM + Prophet (ensemble opcional) | Cobertura de enfoques: tree-based + time series |
| Hiperparámetros | Optuna (Bayesian, 50 trials) | Más eficiente que GridSearch |
| Entorno Python | conda sis-demand (Python 3.10) | Evita conflictos con Anaconda base (NumPy 2.x) |

## Notas de Implementación Importantes

- **Columna plan de seguro:** se llama `PLAN_SEGURO` en los CSV reales (no `PLAN_DE_SEGURO`)
- **Servicios:** 65 tipos distintos (no 32); el mapeo MAPA_SERVICIO de SKILLS.md cubre los principales
- **RAM del equipo: 11.9 GB total** — NUNCA cargar el parquet completo (16.5 GB) en un notebook de EDA
- **Estrategia de carga para EDA:** usar `pq.ParquetFile(...).read_row_groups([...])` con 10-12 grupos → ~1.2M filas → ~400 MB
- **Notebooks de modelado (NB04–NB07):** cargar solo las columnas necesarias con `pd.read_parquet(..., columns=[...])`
- **pandas 2.x:** usar `groupby(..., observed=True)` para evitar FutureWarning con columnas category
- **Min ATENCIONES = 1:** los archivos fuente ya no contienen registros con 0 atenciones

## Feature Engineering

| Feature | Descripción | Estado |
|---|---|---|
| SEMESTRE | 1 o 2 según MES | ✅ Generado en NB00 |
| PERIODO_NUM | Índice temporal numérico (1=2021S1, 2=2021S2, ...) | ✅ Generado en NB00 |
| LOG_ATENCIONES | log1p(ATENCIONES) | Pendiente NB04 |
| SERVICIO_CATEGORIA | Agrupación 65→6 macro-categorías | Pendiente NB04 |
| REGION_TIPO | Costa/Sierra/Selva | Pendiente NB04 |
| LAG_ATENCIONES_1SEM | Atenciones semestre anterior (misma combinación) | Pendiente NB04 ← MUY IMPORTANTE |
| ROLLING_MEAN_2SEM | Media últimos 2 semestres | Pendiente NB04 |
| TENDENCIA_LINEAL | Pendiente de regresión simple histórica | Pendiente NB04 |
| TASA_COBERTURA | ATENCIONES / POBLACION_INEI | Pendiente NB04 (requiere INEI) |

## Estado del Proyecto

| Etapa | Status | Notas |
|---|---|---|
| 00 - Ingesta consolidación | ✅ COMPLETADO | 42M filas, 9 semestres, parquet 353.7M atenciones |
| 01 - EDA Univariado | ✅ COMPLETADO | Skewness confirmada, log1p validado, 65 servicios, tendencia +67% |
| 02 - EDA Bivariado | ✅ COMPLETADO | KW sig. todos los grupos; NIVEL+SERVICIO features más importantes |
| 03 - EDA Multivariado | ✅ COMPLETADO | PCA, VIF, cold-start, clustering K=4 exploratorio |
| 04 - Limpieza + FE | ✅ COMPLETADO | Genera sis_features.parquet con LAG, rolling, tendencia, N_PERIODOS |
| 05 - Clustering IPRESS | ✅ COMPLETADO | K-Means K óptimo, Silhouette + Davies-Bouldin, PCA 2D |
| 06 - Modelado | ✅ COMPLETADO | RF + XGBoost(Optuna 10 trials) + LGBM — MAPE test ~54%, R²~0.31 |
| 07 - Evaluación + SHAP | ✅ COMPLETADO | SHAP con LightGBM (XGB 2.x incompatible con SHAP 0.49) |
| 08 - Cuantificación valor | ✅ COMPLETADO | Dashboard ejecutivo, análisis de sensibilidad |
| App Streamlit | ✅ COMPLETADO | 4 páginas: Dashboard, Predicción, Clustering, Evaluación |

## Issues Conocidos

1. **Encoding real:** UTF-8 (no latin-1). La función `leer_y_estandarizar` ya lo detecta automáticamente.
2. **Cold start problem:** IPRESS nuevas sin historial → usar promedio de IPRESS del mismo tipo/región.
3. **Granularidad temporal:** Solo semestral — no hacer predicciones mensuales.
4. **Columna plan:** `PLAN_SEGURO` (no `PLAN_DE_SEGURO`) — actualizar en todos los notebooks.
5. **pandas 2.x groupby:** agregar `observed=True` siempre que se agrupen columnas category.
6. **RAM EDA:** con 42M filas, usar `.sample(100_000, random_state=42)` para plots y profiling pesado.

## Métricas Reales Obtenidas (NB06 completado 2026-04-26)

| Modelo | MAPE Test | R² Test |
|---|---|---|
| Random Forest | 54.79% | 0.300 |
| XGBoost (Optuna) | 54.26% | 0.297 |
| **LightGBM (mejor)** | **53.96%** | **0.309** |

- **MAPE objetivo era ≤20%** — resultado real 53.96% (granularidad IPRESS×SERVICIO×EDAD×SEXO muy fina)
- **LAG correlaciona 0.747 con ATENCIONES** — el feature es válido pero la heterogeneidad del dataset es muy alta
- **Top feature SHAP: ROLLING_MEAN_2SEM** (0.35), luego TENDENCIA (0.11), N_PERIODOS (0.077)
- **LAG_1SEM en puesto #9** (0.023) — ROLLING_MEAN absorbe su información siendo más estable
- **Mejora vs baseline**: los modelos superan levemente al promedio histórico por NIVEL×EDAD×SEXO
- **Para el TAF:** presentar MAPE ~54% como resultado honesto; contextualizar con la dificultad del problema (datos semestrales, ~8725 IPRESS, 65 servicios, alta variabilidad estocástica)
