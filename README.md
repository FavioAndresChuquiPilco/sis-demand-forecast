# 🏥 SIS-DEMAND: Pronóstico de Demanda de Atenciones Médicas
### Seguro Integral de Salud — Perú | Proyecto de Ciencia de Datos

---

## Problema

El SIS asigna recursos (personal, farmacia, presupuesto) de manera histórica y uniforme, **sin anticipar variaciones en la demanda** de atenciones por establecimiento, servicio y perfil poblacional — generando subutilización en algunos centros de salud y saturación en otros.

## Objetivo

> Construir un modelo predictivo que estime el volumen de atenciones (ATENCIONES) por combinación **[IPRESS × SERVICIO × GRUPO_EDAD × SEXO]** para el siguiente semestre, con **MAPE ≤ 20%**, usando datos históricos del SIS 2021–2024.

## Stack Tecnológico

| Categoría | Herramientas |
|---|---|
| Datos | pandas, numpy, pyarrow (parquet) |
| EDA | ydata-profiling, plotly, scipy, statsmodels |
| ML | scikit-learn, XGBoost, LightGBM, Prophet |
| Optimización | Optuna (Bayesian hyperparameter search) |
| Interpretabilidad | SHAP |
| Clustering | scikit-learn (K-Means) |
| Dashboard | Streamlit + Plotly |

## Instalación

```bash
git clone <repo>
cd sis-demand-forecast
pip install -r requirements.txt
```

## Estructura del Proyecto

```
sis-demand-forecast/
├── data/
│   ├── raw/          # Archivos .xlsx descargados de datos.gob.pe
│   ├── external/     # Datos INEI de población por distrito
│   └── processed/    # .parquet generados por notebooks
├── notebooks/        # Pipeline completo (00–08)
├── src/              # Módulos Python reutilizables
├── app/              # Dashboard Streamlit
├── models/           # Modelos serializados (.pkl)
├── reports/          # Reportes y visualizaciones
├── MEMORY.md         # Contexto del proyecto
├── CLAUDE.md         # Instrucciones para Claude
├── SKILLS.md         # Pipeline técnico detallado
└── requirements.txt
```

## Ejecución del Pipeline

```bash
# 1. Colocar los 9 archivos .xlsx en data/raw/
# 2. Ejecutar notebooks en orden:
jupyter notebook notebooks/00_ingesta_consolidacion.ipynb
# ... continuar con 01, 02, 03, 04, 05, 06, 07, 08

# 3. Lanzar dashboard:
streamlit run app/app.py
```

## Fuentes de Datos

- **SIS Atenciones:** https://www.datosabiertos.gob.pe/dataset/atenciones-realizadas-los-asegurados-del-seguro-integral-de-salud-sis
- **INEI Población:** https://www.inei.gob.pe/estadisticas/indice-tematico/poblacion-y-vivienda/

## Propuesta de Valor

Con una mejora del MAPE del 35% (baseline) al 15% (modelo):
- **~S/. 10 millones anuales** en optimización de recursos humanos
- Reducción de saturación en IPRESS críticas
- Focalización de campañas preventivas en zonas de mayor déficit

---

*Proyecto TAF — Curso de Analítica de Datos | Datos Abiertos SIS — Gobierno del Perú*
