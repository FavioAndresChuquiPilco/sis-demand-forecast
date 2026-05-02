# SKILLS.md — Pipeline Técnico Detallado
## SIS-DEMAND: Pronóstico de Demanda de Atenciones SIS Perú

Este documento define el pipeline completo de Ciencia de Datos, etapa por etapa, con el código de referencia y los outputs esperados de cada fase.

---

## ETAPA 00: Ingesta y Consolidación

**Notebook:** `notebooks/00_ingesta_consolidacion.ipynb`  
**Input:** 9 archivos .xlsx de datos.gob.pe  
**Output:** `data/processed/sis_consolidado.parquet`

### Acciones Clave
```python
import pandas as pd
import numpy as np
from pathlib import Path
from unidecode import unidecode

SEMESTRES = {
    '2021_S1': 1, '2021_S2': 2,
    '2022_S1': 3, '2022_S2': 4,
    '2023_S1': 5, '2023_S2': 6,
    '2024_S1': 7, '2024_S2': 8,
    '2025_S1': 9
}

def leer_y_estandarizar(path: str, semestre_label: str) -> pd.DataFrame:
    """Carga un semestre SIS y lo estandariza."""
    # Intentar múltiples encodings
    for enc in ['latin-1', 'utf-8', 'utf-8-sig']:
        try:
            df = pd.read_excel(path, dtype=str, engine='openpyxl')
            break
        except Exception:
            continue
    
    # Normalizar nombres de columnas
    df.columns = [unidecode(c).upper().strip().replace(' ', '_') for c in df.columns]
    
    # Casteos obligatorios
    df['ATENCIONES'] = pd.to_numeric(df['ATENCIONES'], errors='coerce').fillna(0).astype('int32')
    df['ANO'] = pd.to_numeric(df.get('ANO', df.get('AO', 0)), errors='coerce').astype('int16')
    df['MES'] = pd.to_numeric(df['MES'], errors='coerce').astype('int8')
    df['UBIGEO_DISTRITO'] = df['UBIGEO_DISTRITO'].str.strip().str.zfill(6)
    
    # Limpiar strings
    for col in ['REGION', 'PROVINCIA', 'DISTRITO', 'DESC_SERVICIO', 'DESC_UNIDAD_EJECUTORA']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: unidecode(str(x)).upper().strip())
    
    # Agregar identificador de semestre
    df['SEMESTRE_LABEL'] = semestre_label
    df['PERIODO_NUM'] = SEMESTRES[semestre_label]
    df['SEMESTRE'] = 1 if 'S1' in semestre_label else 2
    
    return df

# Consolidar todos los semestres
dfs = []
for label, num in SEMESTRES.items():
    path = f'data/raw/{label}.xlsx'
    if Path(path).exists():
        dfs.append(leer_y_estandarizar(path, label))

df_raw = pd.concat(dfs, ignore_index=True)

# Validaciones
assert df_raw['ATENCIONES'].min() >= 0, "Atenciones negativas detectadas"
assert df_raw['UBIGEO_DISTRITO'].str.len().max() == 6, "Ubigeos mal formados"
print(f"Dataset consolidado: {df_raw.shape[0]:,} filas, {df_raw.shape[1]} columnas")
print(f"Periodos: {df_raw['PERIODO_NUM'].min()} → {df_raw['PERIODO_NUM'].max()}")

df_raw.to_parquet('data/processed/sis_consolidado.parquet', index=False, compression='snappy')
```

### Validaciones Post-Carga
- [ ] No hay valores nulos en columnas clave (ATENCIONES, REGION, UBIGEO_DISTRITO, DESC_SERVICIO)
- [ ] ATENCIONES >= 0 en todos los registros
- [ ] 9 periodos distintos en PERIODO_NUM
- [ ] UBIGEO_DISTRITO siempre tiene 6 dígitos

---

## ETAPA 01: EDA Univariado

**Notebook:** `notebooks/01_eda_univariado.ipynb`  
**Input:** `data/processed/sis_consolidado.parquet`  
**Output:** Visualizaciones + `reports/eda_univariado_summary.csv`

### Análisis de Variable Objetivo (ATENCIONES)
```python
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

df = pd.read_parquet('data/processed/sis_consolidado.parquet')

# 1. Estadísticas descriptivas
desc = df['ATENCIONES'].describe(percentiles=[.25, .50, .75, .90, .95, .99])

# 2. Test de normalidad (sobre muestra si el dataset es grande)
sample = df['ATENCIONES'].sample(min(5000, len(df)), random_state=42)
stat, p_value = stats.kstest(sample, 'norm', 
                              args=(sample.mean(), sample.std()))
print(f"Kolmogorov-Smirnov: stat={stat:.4f}, p={p_value:.4e}")
# Esperado: p << 0.05 → NO es normal → confirma necesidad de log-transform

# 3. Distribución con y sin log-transform
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['ATENCIONES'], name='Original', nbinsx=50))
fig.add_trace(go.Histogram(x=np.log1p(df['ATENCIONES']), name='log1p', nbinsx=50))
fig.show()

# 4. Skewness y Kurtosis
print(f"Skewness: {df['ATENCIONES'].skew():.2f}")  # Esperado: > 2
print(f"Kurtosis: {df['ATENCIONES'].kurtosis():.2f}")

# 5. Q-Q Plot
stats.probplot(np.log1p(sample), dist="norm", plot=plt)
```

### Análisis de Variables Categóricas
```python
# Para cada variable categórica: frecuencia relativa + gráfico de barras
cat_vars = ['REGION', 'NIVEL_EESS', 'PLAN_DE_SEGURO', 'SEXO', 'GRUPO_EDAD',
            'DESC_SERVICIO', 'SEMESTRE']

for var in cat_vars:
    freq = df[var].value_counts(normalize=True).reset_index()
    freq.columns = [var, 'PROPORCION']
    print(f"\n{var} ({df[var].nunique()} categorías):")
    print(freq.head(10).to_string(index=False))
```

### Reporte Automático
```python
from ydata_profiling import ProfileReport
profile = ProfileReport(df.sample(50000, random_state=42), title="SIS EDA Report")
profile.to_file("reports/eda_profile_report.html")
```

---

## ETAPA 02: EDA Bivariado

**Notebook:** `notebooks/02_eda_bivariado.ipynb`

### Análisis ATENCIONES vs Variables Clave
```python
# Test Kruskal-Wallis (no paramétrico) para diferencias entre grupos
from scipy.stats import kruskal

# H1: NIVEL_EESS diferente → esperado: p << 0.05
grupos = [df[df['NIVEL_EESS']==n]['ATENCIONES'].values for n in df['NIVEL_EESS'].unique()]
stat, p = kruskal(*grupos)
print(f"Kruskal-Wallis NIVEL_EESS: H={stat:.2f}, p={p:.4e}")

# H2: Estacionalidad por semestre
stat, p = kruskal(
    df[df['SEMESTRE']==1]['ATENCIONES'].values,
    df[df['SEMESTRE']==2]['ATENCIONES'].values
)
print(f"Kruskal-Wallis Semestre 1 vs 2: H={stat:.2f}, p={p:.4e}")

# Serie de tiempo: atenciones totales por semestre a nivel nacional
ts = df.groupby('PERIODO_NUM')['ATENCIONES'].sum().reset_index()
fig = px.line(ts, x='PERIODO_NUM', y='ATENCIONES', 
              title='Tendencia Nacional de Atenciones SIS 2021-2025',
              labels={'PERIODO_NUM': 'Semestre', 'ATENCIONES': 'Total Atenciones'})
fig.show()

# Heatmap REGION × DESC_SERVICIO
pivot = df.pivot_table(values='ATENCIONES', index='REGION', 
                        columns='DESC_SERVICIO', aggfunc='sum', fill_value=0)
fig = px.imshow(pivot, title='Demanda por Región y Tipo de Servicio', 
                color_continuous_scale='Blues')
fig.show()
```

### Correlaciones con Encoding Básico
```python
# Encoding ordinal básico para correlaciones
from sklearn.preprocessing import LabelEncoder

df_enc = df.copy()
for col in ['REGION', 'NIVEL_EESS', 'DESC_SERVICIO', 'GRUPO_EDAD', 'SEXO']:
    df_enc[col + '_ENC'] = LabelEncoder().fit_transform(df_enc[col].fillna('UNKNOWN'))

num_cols = [c for c in df_enc.columns if c.endswith('_ENC')] + ['ATENCIONES', 'PERIODO_NUM']
corr_matrix = df_enc[num_cols].corr(method='spearman')  # Spearman para datos no normales

fig = px.imshow(corr_matrix, title='Correlaciones de Spearman', 
                color_continuous_scale='RdBu', range_color=[-1, 1])
fig.show()
```

---

## ETAPA 03: EDA Multivariado

**Notebook:** `notebooks/03_eda_multivariado.ipynb`

```python
# PCA exploratorio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Usar dataset reducido para PCA
df_pca = df.groupby(['REGION', 'DESC_SERVICIO', 'GRUPO_EDAD', 'SEXO', 'NIVEL_EESS', 'PERIODO_NUM'])[
    'ATENCIONES'].sum().reset_index()

# Encoding y escalado
features_pca = ['REGION_ENC', 'SERVICIO_ENC', 'EDAD_ENC', 'SEXO_ENC', 'NIVEL_ENC', 'PERIODO_NUM']
X_scaled = StandardScaler().fit_transform(df_pca[features_pca])

pca = PCA(n_components=5, random_state=42)
pca.fit(X_scaled)

# Varianza explicada
print("Varianza explicada por componente:", pca.explained_variance_ratio_.round(3))
print("Varianza acumulada:", np.cumsum(pca.explained_variance_ratio_).round(3))

# VIF para multicolinealidad
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame({'feature': features_pca,
    'VIF': [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]})
print(vif_data.sort_values('VIF', ascending=False))
```

---

## ETAPA 04: Limpieza y Feature Engineering

**Notebook:** `notebooks/04_limpieza_feature_engineering.ipynb`  
**Input:** `data/processed/sis_consolidado.parquet`  
**Output:** `data/processed/sis_features.parquet`

```python
df = pd.read_parquet('data/processed/sis_consolidado.parquet')

# ==================== LIMPIEZA ====================

# 1. Winsorización de ATENCIONES (por nivel + servicio para no perder señal real)
df['ATENCIONES_ORIGINAL'] = df['ATENCIONES'].copy()
p99 = df.groupby(['NIVEL_EESS', 'DESC_SERVICIO'])['ATENCIONES'].transform(lambda x: x.quantile(0.99))
df['ATENCIONES'] = df['ATENCIONES'].clip(upper=p99)

# 2. Eliminar registros con ATENCIONES = 0 (no aportan señal)
df = df[df['ATENCIONES'] > 0].reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================

# Target transformado
df['LOG_ATENCIONES'] = np.log1p(df['ATENCIONES'])

# Macro-categorías de servicio
MAPA_SERVICIO = {
    'CONSULTA EXTERNA': 'CURATIVO',
    'ATENCION POR EMERGENCIA': 'URGENCIAS',
    'ATENCION POR EMERGENCIA CON OBSERVACION': 'URGENCIAS',
    'INTERNAMIENTO EN EESS SIN INTERVENCION QUIRURGICA': 'URGENCIAS',
    'ATENCION PRENATAL': 'MATERNO',
    'SALUD REPRODUCTIVA (PLANIFICACION FAMILIAR)': 'MATERNO',
    'CESAREA': 'MATERNO',
    'DIAGNOSTICO DEL EMBARAZO': 'MATERNO',
    'ATENCION DE PARTO VAGINAL': 'MATERNO',
    'ATENCION DEL PUERPERIO NORMAL': 'MATERNO',
    'EXAMENES DE ECOGRAFIA OBSTETRICA': 'MATERNO',
    'EXAMENES DE LABORATORIO COMPLETO DE LA GESTANTE': 'MATERNO',
    'ATENCION PRECONCEPCIONAL': 'MATERNO',
    'CONTROL DE CRECIMIENTO Y DESARROLLO EN MENORES DE 0 - 4 ANOS': 'PEDIATRICO',
    'CONTROL DE CRECIMIENTO Y DESARROLLO EN MENORES DE 5 - 11 ANOS': 'PEDIATRICO',
    'ESTIMULACION TEMPRANA PARA MENORES DE 36 MESES': 'PEDIATRICO',
    'ATENCION INMEDIATA DEL RECIEN NACIDO NORMAL': 'PEDIATRICO',
    'SUPLEMENTO DE MICRONUTRIENTES': 'PEDIATRICO',
    'PROFILAXIS ANTIPARASITARIA': 'PREVENTIVO',
    'APOYO AL DIAGNOSTICO': 'PREVENTIVO',
    'DETECCION DE PROBLEMAS EN SALUD MENTAL': 'PREVENTIVO',
    'SALUD BUCAL': 'PREVENTIVO',
    'ATENCION INTEGRAL DEL ADOLESCENTE': 'PREVENTIVO',
    'ATENCION INTEGRAL DE SALUD DEL JOVEN Y ADULTO': 'PREVENTIVO',
    'ATENCION EXTRAMURAL RURAL (VISITA DOMICILIARIA)': 'PREVENTIVO',
    'DETECCION TRASTORNO AGUDEZA VISUAL Y CEGUERA': 'PREVENTIVO',
    'TRATAMIENTO DE ITS EN ADOLESCENTES, ADULTOS Y ADULTOS MAYORES': 'PREVENTIVO',
    'TELEMONITOREO CON PRESCRIPCION Y ENTREGA DE MEDICAMENTOS': 'TELEMATICA',
    'TELEORIENTACION CON PRESCRIPCION Y ENTREGA DE MEDICAMENTOS': 'TELEMATICA',
}
df['SERVICIO_CATEGORIA'] = df['DESC_SERVICIO'].map(MAPA_SERVICIO).fillna('OTROS')

# Tipo geográfico por región
MAPA_REGION_TIPO = {
    'LIMA': 'COSTA', 'CALLAO': 'COSTA', 'LA LIBERTAD': 'COSTA', 'PIURA': 'COSTA',
    'LAMBAYEQUE': 'COSTA', 'ICA': 'COSTA', 'AREQUIPA': 'COSTA', 'TACNA': 'COSTA',
    'TUMBES': 'COSTA', 'MOQUEGUA': 'COSTA', 'ANCASH': 'SIERRA', 'CAJAMARCA': 'SIERRA',
    'CUSCO': 'SIERRA', 'PUNO': 'SIERRA', 'JUNIN': 'SIERRA', 'AYACUCHO': 'SIERRA',
    'APURIMAC': 'SIERRA', 'HUANCAVELICA': 'SIERRA', 'HUANUCO': 'SIERRA', 'PASCO': 'SIERRA',
    'AMAZONAS': 'SELVA', 'LORETO': 'SELVA', 'UCAYALI': 'SELVA', 'MADRE DE DIOS': 'SELVA',
    'SAN MARTIN': 'SELVA',
}
df['REGION_TIPO'] = df['REGION'].map(MAPA_REGION_TIPO).fillna('COSTA')

# Encoding ordinal para GRUPO_EDAD
ORDEN_EDAD = {'00 - 04 ANOS': 0, '05 - 11 ANOS': 1, '12 - 17 ANOS': 2,
              '18 - 29 ANOS': 3, '30 - 59 ANOS': 4, '60 - MAS ANOS': 5}
df['GRUPO_EDAD_ORD'] = df['GRUPO_EDAD'].map(ORDEN_EDAD)

# Encoding binario SEXO
df['SEXO_BIN'] = (df['SEXO'] == 'FEMENINO').astype('int8')

# Encoding ordinal NIVEL_EESS
df['NIVEL_NUM'] = df['NIVEL_EESS'].map({'I': 1, 'II': 2, 'III': 3}).fillna(1).astype('int8')

# FLAGS de servicios especiales
df['ES_TELEMATICA'] = (df['SERVICIO_CATEGORIA'] == 'TELEMATICA').astype('int8')
df['ES_PREVENTIVO'] = (df['SERVICIO_CATEGORIA'] == 'PREVENTIVO').astype('int8')
df['ES_MATERNO'] = (df['SERVICIO_CATEGORIA'] == 'MATERNO').astype('int8')

# ==================== FEATURES TEMPORALES (LAG) ====================
# Ordenar para crear lags correctamente
df = df.sort_values(['COD_IPRESS', 'COD_SERVICIO', 'GRUPO_EDAD', 'SEXO', 'PERIODO_NUM'])

# Clave de agrupación para lags
df['KEY'] = df['COD_IPRESS'] + '_' + df['COD_SERVICIO'] + '_' + df['GRUPO_EDAD'] + '_' + df['SEXO']

# LAG 1 semestre
df['LAG_ATENCIONES_1SEM'] = df.groupby('KEY')['ATENCIONES'].shift(1)

# Rolling mean 2 semestres
df['ROLLING_MEAN_2SEM'] = df.groupby('KEY')['ATENCIONES'].transform(
    lambda x: x.shift(1).rolling(2, min_periods=1).mean()
)

# Tendencia lineal simple (pendiente de regresión sobre historial de cada KEY)
def calc_pendiente(series):
    """Calcula pendiente de regresión lineal para capturar tendencia."""
    valid = series.dropna()
    if len(valid) < 2:
        return 0
    x = np.arange(len(valid))
    return np.polyfit(x, valid, 1)[0]

tendencias = df.groupby('KEY')['ATENCIONES'].apply(calc_pendiente).reset_index()
tendencias.columns = ['KEY', 'TENDENCIA_LINEAL']
df = df.merge(tendencias, on='KEY', how='left')

# Target Encoding para REGION (media de ATENCIONES por región — solo en train)
# NOTA: Fit solo en train, aplicar en val/test para evitar leakage
# Esto se hace en el notebook de modelado

# Guardar dataset de features
df.drop(columns=['KEY'], inplace=True)
df.to_parquet('data/processed/sis_features.parquet', index=False, compression='snappy')
print(f"Dataset con features: {df.shape[0]:,} filas, {df.shape[1]} columnas")
```

---

## ETAPA 05: Clustering de IPRESS

**Notebook:** `notebooks/05_clustering_ipress.ipynb`

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Perfil de cada IPRESS en el periodo más reciente
df_ipress = df[df['PERIODO_NUM'] >= 6].groupby('COD_IPRESS').agg(
    TOTAL_ATENCIONES=('ATENCIONES', 'sum'),
    NIVEL=('NIVEL_NUM', 'first'),
    REGION_TIPO=('REGION_TIPO', 'first'),
    PCT_PEDIATRICO=('ES_PREVENTIVO', 'mean'),  # Aproximación
    PCT_MATERNO=('ES_MATERNO', 'mean'),
    PCT_TELEMATICA=('ES_TELEMATICA', 'mean'),
    PCT_FEMENINO=('SEXO_BIN', 'mean'),
    EDAD_MEDIA_ORD=('GRUPO_EDAD_ORD', 'mean'),
).reset_index()

# Escalar
features_cluster = ['TOTAL_ATENCIONES', 'NIVEL', 'PCT_PEDIATRICO', 
                    'PCT_MATERNO', 'PCT_TELEMATICA', 'PCT_FEMENINO', 'EDAD_MEDIA_ORD']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_ipress[features_cluster])

# Método del codo + Silhouette
inertias, silhouettes = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels))

# Seleccionar k óptimo (generalmente 4-6 para el SIS)
k_optimo = K_range[np.argmax(silhouettes)]
print(f"K óptimo por Silhouette: {k_optimo} (score: {max(silhouettes):.3f})")

# Entrenar modelo final
km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_ipress['CLUSTER'] = km_final.fit_predict(X_cluster)

# Perfil de cada cluster
print(df_ipress.groupby('CLUSTER')[features_cluster].mean().round(2))
```

---

## ETAPA 06: Modelado Predictivo

**Notebook:** `notebooks/06_modelado_predictivo.ipynb`

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# Separación temporal ESTRICTA
df = pd.read_parquet('data/processed/sis_features.parquet')
df = df.dropna(subset=['LAG_ATENCIONES_1SEM'])  # Requiere al menos 1 semestre anterior

FEATURES = [
    'NIVEL_NUM', 'GRUPO_EDAD_ORD', 'SEXO_BIN', 'SEMESTRE', 'PERIODO_NUM',
    'ES_TELEMATICA', 'ES_PREVENTIVO', 'ES_MATERNO',
    'LAG_ATENCIONES_1SEM', 'ROLLING_MEAN_2SEM', 'TENDENCIA_LINEAL',
    'REGION_ENC',        # Target encoding (fit solo en train)
    'SERVICIO_ENC',      # Target encoding
    'REGION_TIPO_ENC',   # Label encoding
]
TARGET = 'LOG_ATENCIONES'

# Splits
train = df[df['PERIODO_NUM'] <= 6]
val   = df[df['PERIODO_NUM'] == 7]
test  = df[df['PERIODO_NUM'] == 8]

# Target Encoding — FIT SOLO EN TRAIN
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['REGION', 'DESC_SERVICIO', 'REGION_TIPO'])
X_train = te.fit_transform(train[['REGION','DESC_SERVICIO','REGION_TIPO'] + FEATURES[:-3]], train[TARGET])
X_val   = te.transform(val[['REGION','DESC_SERVICIO','REGION_TIPO'] + FEATURES[:-3]])
X_test  = te.transform(test[['REGION','DESC_SERVICIO','REGION_TIPO'] + FEATURES[:-3]])

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

# ===== BASELINE: Promedio histórico =====
baseline_pred = val.groupby('KEY')['ATENCIONES'].transform('mean')
# (usando promedio histórico de cada combinación en train)

# ===== MODELO 1: Random Forest =====
rf = RandomForestRegressor(n_estimators=300, max_depth=12, 
                            min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ===== MODELO 2: XGBoost (con Optuna) =====
def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=50, show_progress_bar=True)
xgb_best = XGBRegressor(**study.best_params, random_state=42)
xgb_best.fit(X_train, y_train)

# ===== MODELO 3: LightGBM =====
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, 
                      max_depth=8, num_leaves=63, random_state=42)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

---

## ETAPA 07: Evaluación e Interpretabilidad

**Notebook:** `notebooks/07_evaluacion_interpretabilidad.ipynb`

```python
import shap

def evaluar_modelo(nombre, model, X, y_true_log):
    """Evalúa un modelo y retorna métricas en escala original."""
    pred_log = model.predict(X)
    pred_orig = np.expm1(pred_log)
    true_orig = np.expm1(y_true_log)
    
    mae  = mean_absolute_error(true_orig, pred_orig)
    rmse = mean_squared_error(true_orig, pred_orig, squared=False)
    mape = np.mean(np.abs((true_orig - pred_orig) / (true_orig + 1))) * 100
    r2   = r2_score(true_orig, pred_orig)
    
    print(f"\n{'='*40}")
    print(f"Modelo: {nombre}")
    print(f"  MAE  = {mae:.2f} atenciones")
    print(f"  RMSE = {rmse:.2f} atenciones")
    print(f"  MAPE = {mape:.2f}%")
    print(f"  R²   = {r2:.4f}")
    return {'modelo': nombre, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

resultados = []
for nombre, modelo in [('RandomForest', rf), ('XGBoost', xgb_best), ('LightGBM', lgbm)]:
    resultados.append(evaluar_modelo(nombre, modelo, X_test, y_test))

# Seleccionar mejor modelo
df_resultados = pd.DataFrame(resultados).sort_values('MAPE')
mejor_modelo_nombre = df_resultados.iloc[0]['modelo']

# ===== SHAP Analysis con mejor modelo =====
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test.sample(min(5000, len(X_test)), random_state=42))

# Summary plot (importancia global)
shap.summary_plot(shap_values, X_test.sample(min(5000, len(X_test)), random_state=42),
                  plot_type='bar', max_display=15)

# Dependence plot para LAG (el feature más importante esperado)
shap.dependence_plot('LAG_ATENCIONES_1SEM', shap_values, 
                     X_test.sample(min(5000, len(X_test)), random_state=42))
```

---

## ETAPA 08: Cuantificación de Valor

**Notebook:** `notebooks/08_propuesta_valor_cuantificada.ipynb`

```python
# ===== PROPUESTA DE VALOR CUANTIFICADA =====
# Metodología: Costo de Ineficiencia Evitada

# Supuestos basados en MINSA y literatura de salud pública peruana
COSTO_MEDICO_MES = 8_500       # S/. por médico/mes en establecimiento Nivel I (promedio SIS)
TOTAL_IPRESS_SIS = 7_800       # Total IPRESS activas en la red SIS (aprox)
IMPACTO_MEJORA_PCT = 0.10      # Asumimos impacto en 10% de IPRESS
ERROR_BASELINE_MAPE = 0.35     # MAPE baseline (promedio histórico simple): ~35%
ERROR_MODELO_MAPE   = 0.15     # MAPE objetivo de nuestro modelo: ~15%

mejora_relativa = (ERROR_BASELINE_MAPE - ERROR_MODELO_MAPE) / ERROR_BASELINE_MAPE
print(f"Mejora relativa en error de planificación: {mejora_relativa:.1%}")

ipress_impactadas = TOTAL_IPRESS_SIS * IMPACTO_MEJORA_PCT
ahorro_mensual = ipress_impactadas * COSTO_MEDICO_MES * mejora_relativa
ahorro_anual = ahorro_mensual * 12

print(f"\nRESUMEN DE PROPUESTA DE VALOR:")
print(f"  IPRESS impactadas: {ipress_impactadas:,.0f}")
print(f"  Ahorro mensual estimado: S/. {ahorro_mensual:,.0f}")
print(f"  Ahorro anual estimado: S/. {ahorro_anual:,.0f}")
print(f"  ROI estimado (vs costo proyecto): >> 10x")

# Impacto adicional: reducción de atenciones perdidas
# Si reducimos saturación en IPRESS de alta demanda → menos pacientes sin atender
```

---

## STREAMLIT APP — Guía de Implementación

```python
# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.set_page_config(page_title="🏥 SIS Demand Forecast", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet('data/processed/sis_features.parquet')

@st.cache_resource
def load_model():
    with open('models/xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

# === SIDEBAR ===
with st.sidebar:
    st.image("assets/sis_logo.png", width=150)
    st.title("SIS Demand Forecast")
    pagina = st.selectbox("Navegación", 
        ["🏠 Dashboard Nacional", "🔮 Predicción de Demanda", 
         "🗺️ Segmentación IPRESS", "⚠️ Alertas Tempranas"])

# === PÁGINA 1: DASHBOARD NACIONAL ===
if pagina == "🏠 Dashboard Nacional":
    st.title("Dashboard Nacional — Atenciones SIS")
    
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Atenciones (2024)", f"{df[df['PERIODO_NUM']==8]['ATENCIONES'].sum():,.0f}")
    with col2:
        st.metric("IPRESS Activas", f"{df['COD_IPRESS'].nunique():,}")
    with col3:
        st.metric("Regiones Cubiertas", f"{df['REGION'].nunique()}")
    with col4:
        st.metric("Servicios Distintos", f"{df['DESC_SERVICIO'].nunique()}")
    
    # Mapa de calor por región
    df_region = df.groupby('REGION')['ATENCIONES'].sum().reset_index()
    fig = px.bar(df_region.sort_values('ATENCIONES', ascending=False), 
                 x='REGION', y='ATENCIONES', title='Atenciones por Región',
                 color='ATENCIONES', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# === PÁGINA 2: PREDICCIÓN ===
elif pagina == "🔮 Predicción de Demanda":
    st.title("🔮 Predicción de Demanda de Atenciones")
    
    col1, col2 = st.columns(2)
    with col1:
        region_sel = st.selectbox("Región", sorted(df['REGION'].unique()))
        servicio_sel = st.selectbox("Servicio", sorted(df['DESC_SERVICIO'].unique()))
    with col2:
        edad_sel = st.selectbox("Grupo Edad", df['GRUPO_EDAD'].unique())
        semestre_pred = st.slider("Semestre a predecir", 1, 3, 1)
    
    if st.button("🚀 Generar Predicción"):
        # Lógica de predicción aquí
        st.success(f"Demanda proyectada para {region_sel} — {servicio_sel}: **X atenciones**")
```

---

## Checklist Final del Proyecto

- [ ] 9 semestres de datos descargados y consolidados en parquet
- [ ] EDA completo con tests estadísticos (Kruskal-Wallis, Spearman, KS-test)
- [ ] Feature engineering completo (LAGs, rolling, tendencia, categorías)
- [ ] Join con datos INEI (tasa de cobertura)
- [ ] Clustering K-Means con Silhouette Score óptimo
- [ ] 3 modelos entrenados (RF, XGBoost, LightGBM) + baseline
- [ ] Evaluación con MAE, RMSE, MAPE, R² en conjunto de test temporal
- [ ] SHAP values con top-10 features
- [ ] Cuantificación monetaria del impacto (S/.)
- [ ] Dashboard Streamlit con 4 páginas funcionales
- [ ] Documento TAF con todas las secciones requeridas
- [ ] Video de 6-8 minutos con data storytelling
