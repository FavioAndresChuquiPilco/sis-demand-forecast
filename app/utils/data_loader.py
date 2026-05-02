"""Carga centralizada y cacheada — usa datos pre-agregados para el deploy."""
import pandas as pd
import pickle
import streamlit as st
from pathlib import Path

BASE    = Path(__file__).parent.parent.parent   # raiz del repo
DASH    = BASE / "data" / "dashboard"
PROC    = BASE / "data" / "processed"
MODELS  = BASE / "models"
REPORTS = BASE / "reports"
OUTPUTS = BASE / "data" / "outputs"

# ── Datos pre-agregados (ligeros, < 1 MB total) ──────────────────

@st.cache_data(show_spinner=False)
def cargar_ts_nacional():
    return pd.read_parquet(DASH / "ts_nacional.parquet")

@st.cache_data(show_spinner=False)
def cargar_ts_region():
    return pd.read_parquet(DASH / "ts_region.parquet")

@st.cache_data(show_spinner=False)
def cargar_ts_servicio():
    return pd.read_parquet(DASH / "ts_servicio.parquet")

@st.cache_data(show_spinner=False)
def cargar_dist_nivel():
    return pd.read_parquet(DASH / "dist_nivel.parquet")

@st.cache_data(show_spinner=False)
def cargar_dist_edad():
    return pd.read_parquet(DASH / "dist_edad.parquet")

@st.cache_data(show_spinner=False)
def cargar_top_servicios():
    return pd.read_parquet(DASH / "top_servicios.parquet")

@st.cache_data(show_spinner=False)
def cargar_top_regiones():
    return pd.read_parquet(DASH / "top_regiones.parquet")

@st.cache_data(show_spinner=False)
def cargar_catalogos():
    df = pd.read_parquet(DASH / "catalogos.parquet")
    return {
        "regiones":  df[df["tipo"] == "region"]["valor"].tolist(),
        "servicios": df[df["tipo"] == "servicio"]["valor"].tolist(),
        "edades":    df[df["tipo"] == "edad"]["valor"].tolist(),
        "niveles":   df[df["tipo"] == "nivel"]["valor"].tolist(),
        "sexos":     df[df["tipo"] == "sexo"]["valor"].tolist(),
    }

# ── Modelos ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    with open(MODELS / "lgbm_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def cargar_encoder():
    with open(MODELS / "target_encoder.pkl", "rb") as f:
        return pickle.load(f)

# ── Clusters ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_clusters():
    return pd.read_parquet(PROC / "ipress_clusters.parquet")

# ── Reports ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_resultados_modelos():
    try:
        return pd.read_csv(REPORTS / "resultados_modelos.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def cargar_eval_errores():
    try:
        dist  = pd.read_parquet(DASH / "eval_error_dist.parquet")
        nivel = pd.read_parquet(DASH / "eval_error_nivel.parquet")
        return dist, nivel
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner=False)
def cargar_propuesta_valor():
    try:
        return pd.read_csv(REPORTS / "propuesta_valor.csv")
    except Exception:
        return pd.DataFrame()

# ── Constantes del universo completo (NB00) ──────────────────────

TOTAL_FILAS      = 41_997_568
TOTAL_ATENCIONES = 353_696_545
N_IPRESS         = 8_725
N_REGIONES       = 26
N_SERVICIOS      = 65

ETIQUETAS_PERIODO = {
    1:"2021 S1", 2:"2021 S2", 3:"2022 S1", 4:"2022 S2",
    5:"2023 S1", 6:"2023 S2", 7:"2024 S1", 8:"2024 S2", 9:"2025 S1",
}

ORDEN_EDAD = [
    "00 - 04 ANOS","05 - 11 ANOS","12 - 17 ANOS",
    "18 - 29 ANOS","30 - 59 ANOS","60 - MAS ANOS",
]
