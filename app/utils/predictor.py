"""Pipeline de predicción para la app — mismo encoding que NB06."""
import numpy as np
import pandas as pd

FEATURES = [
    "NIVEL_NUM", "GRUPO_EDAD_ORD", "SEXO_BIN", "SEMESTRE", "PERIODO_NUM",
    "ES_TELEMATICA", "ES_PREVENTIVO", "ES_MATERNO",
    "LAG_ATENCIONES_1SEM", "ROLLING_MEAN_2SEM", "TENDENCIA_LINEAL",
    "N_PERIODOS_ACTIVOS", "CLUSTER",
]
COLS_TE = ["REGION", "DESC_SERVICIO"]
COLS_PASS = ["REGION_TIPO", "SERVICIO_CATEGORIA"]

MAPA_SERVICIO = {
    "CONSULTA EXTERNA": "CURATIVO",
    "ATENCION POR EMERGENCIA": "URGENCIAS",
    "ATENCION POR EMERGENCIA CON OBSERVACION": "URGENCIAS",
    "INTERNAMIENTO EN EESS SIN INTERVENCION QUIRURGICA": "URGENCIAS",
    "ATENCION PRENATAL": "MATERNO",
    "SALUD REPRODUCTIVA (PLANIFICACION FAMILIAR)": "MATERNO",
    "CESAREA": "MATERNO", "DIAGNOSTICO DEL EMBARAZO": "MATERNO",
    "ATENCION DE PARTO VAGINAL": "MATERNO",
    "ATENCION DEL PUERPERIO NORMAL": "MATERNO",
    "CONTROL DE CRECIMIENTO Y DESARROLLO EN MENORES DE 0 - 4 ANOS": "PEDIATRICO",
    "CONTROL DE CRECIMIENTO Y DESARROLLO EN MENORES DE 5 - 11 ANOS": "PEDIATRICO",
    "ESTIMULACION TEMPRANA PARA MENORES DE 36 MESES": "PEDIATRICO",
    "ATENCION INMEDIATA DEL RECIEN NACIDO NORMAL": "PEDIATRICO",
    "SUPLEMENTO DE MICRONUTRIENTES": "PEDIATRICO",
    "PROFILAXIS ANTIPARASITARIA": "PREVENTIVO",
    "APOYO AL DIAGNOSTICO": "PREVENTIVO",
    "DETECCION DE PROBLEMAS EN SALUD MENTAL": "PREVENTIVO",
    "SALUD BUCAL": "PREVENTIVO",
    "ATENCION INTEGRAL DEL ADOLESCENTE": "PREVENTIVO",
    "ATENCION INTEGRAL DE SALUD DEL JOVEN Y ADULTO": "PREVENTIVO",
    "ATENCION EXTRAMURAL RURAL (VISITA DOMICILIARIA)": "PREVENTIVO",
    "TELEMONITOREO CON PRESCRIPCION Y ENTREGA DE MEDICAMENTOS": "TELEMATICA",
    "TELEORIENTACION CON PRESCRIPCION Y ENTREGA DE MEDICAMENTOS": "TELEMATICA",
}

MAPA_REGION_TIPO = {
    "LIMA METROPOLITANA": "COSTA", "LIMA": "COSTA", "CALLAO": "COSTA",
    "LA LIBERTAD": "COSTA", "PIURA": "COSTA", "LAMBAYEQUE": "COSTA",
    "ICA": "COSTA", "AREQUIPA": "COSTA", "TACNA": "COSTA",
    "TUMBES": "COSTA", "MOQUEGUA": "COSTA",
    "ANCASH": "SIERRA", "CAJAMARCA": "SIERRA", "CUSCO": "SIERRA",
    "PUNO": "SIERRA", "JUNIN": "SIERRA", "AYACUCHO": "SIERRA",
    "APURIMAC": "SIERRA", "HUANCAVELICA": "SIERRA",
    "HUANUCO": "SIERRA", "PASCO": "SIERRA",
    "AMAZONAS": "SELVA", "LORETO": "SELVA", "UCAYALI": "SELVA",
    "MADRE DE DIOS": "SELVA", "SAN MARTIN": "SELVA",
}

ORDEN_EDAD_ORD = {
    "00 - 04 ANOS": 0, "05 - 11 ANOS": 1, "12 - 17 ANOS": 2,
    "18 - 29 ANOS": 3, "30 - 59 ANOS": 4, "60 - MAS ANOS": 5,
}

MAPA_NIVEL = {"I": 1, "II": 2, "III": 3}

SERVICIO_CATEGORIAS = {
    "CURATIVO": 0, "URGENCIAS": 1, "MATERNO": 2,
    "PEDIATRICO": 3, "PREVENTIVO": 4, "TELEMATICA": 5, "OTROS": 6,
}
REGION_TIPO_ENC = {"COSTA": 0, "SIERRA": 1, "SELVA": 2}


def predecir(
    modelo, enc_obj,
    region, nivel_eess, desc_servicio, grupo_edad, sexo,
    semestre, periodo_num,
    lag_atenciones, rolling_mean, tendencia,
    n_periodos, cluster,
):
    """Genera una predicción de ATENCIONES para un registro."""
    enc_maps = enc_obj["enc_maps"]
    global_mean = enc_obj["global_mean"]

    # Target encoding
    region_enc = enc_maps.get("REGION", {}).get(region, global_mean)
    servicio_enc = enc_maps.get("DESC_SERVICIO", {}).get(desc_servicio, global_mean)

    # Label encoding
    servicio_cat = MAPA_SERVICIO.get(desc_servicio, "OTROS")
    region_tipo = MAPA_REGION_TIPO.get(region, "COSTA")
    serv_cat_enc = SERVICIO_CATEGORIAS.get(servicio_cat, 6)
    region_tipo_enc = REGION_TIPO_ENC.get(region_tipo, 0)

    # Flags
    es_telematica = 1 if servicio_cat == "TELEMATICA" else 0
    es_preventivo = 1 if servicio_cat == "PREVENTIVO" else 0
    es_materno = 1 if servicio_cat == "MATERNO" else 0

    nivel_num = MAPA_NIVEL.get(nivel_eess, 1)
    edad_ord = ORDEN_EDAD_ORD.get(grupo_edad, 0)
    sexo_bin = 1 if sexo == "FEMENINO" else 0

    fila = pd.DataFrame([{
        "NIVEL_NUM": nivel_num,
        "GRUPO_EDAD_ORD": edad_ord,
        "SEXO_BIN": sexo_bin,
        "SEMESTRE": semestre,
        "PERIODO_NUM": periodo_num,
        "ES_TELEMATICA": es_telematica,
        "ES_PREVENTIVO": es_preventivo,
        "ES_MATERNO": es_materno,
        "LAG_ATENCIONES_1SEM": float(lag_atenciones),
        "ROLLING_MEAN_2SEM": float(rolling_mean),
        "TENDENCIA_LINEAL": float(tendencia),
        "N_PERIODOS_ACTIVOS": int(n_periodos),
        "CLUSTER": int(cluster),
        "REGION": float(region_enc),
        "DESC_SERVICIO": float(servicio_enc),
        "REGION_TIPO": region_tipo_enc,
        "SERVICIO_CATEGORIA": serv_cat_enc,
    }])

    pred_log = modelo.predict(fila)[0]
    pred_orig = np.expm1(pred_log)
    return max(1, round(pred_orig))
