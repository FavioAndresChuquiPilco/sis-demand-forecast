import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    cargar_ts_nacional, cargar_ts_region, cargar_ts_servicio,
    cargar_dist_nivel, cargar_dist_edad, cargar_top_servicios,
    cargar_top_regiones, cargar_catalogos, cargar_modelo,
    cargar_encoder, cargar_clusters, cargar_resultados_modelos,
    cargar_eval_errores, cargar_propuesta_valor,
    ETIQUETAS_PERIODO, ORDEN_EDAD,
    TOTAL_FILAS, TOTAL_ATENCIONES, N_IPRESS, N_REGIONES, N_SERVICIOS,
)
from utils.predictor import predecir

st.set_page_config(
    page_title="SIS Demand Forecast",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 SIS-DEMAND")
    st.caption("Pronóstico de Demanda de Atenciones\nSeguro Integral de Salud — Perú")
    st.divider()
    pagina = st.selectbox(
        "Navegación",
        ["🏠 Dashboard Nacional",
         "🔮 Predicción de Demanda",
         "🗂️ Segmentación de IPRESS",
         "📊 Evaluación del Modelo"],
    )
    st.divider()
    st.caption("TAF — Analítica de Datos\nCENTRUM PUCP · 2026")

# ═════════════════════════════════════════════════════════════════
# PÁGINA 1: DASHBOARD NACIONAL
# ═════════════════════════════════════════════════════════════════
if pagina == "🏠 Dashboard Nacional":
    st.title("Dashboard Nacional — Atenciones SIS 2021–2025")

    ts = cargar_ts_nacional()
    ultimo_p    = int(ts["PERIODO_NUM"].max())
    total_ult   = int(ts[ts["PERIODO_NUM"] == ultimo_p]["ATENCIONES"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Atenciones {ETIQUETAS_PERIODO[ultimo_p]}", f"{total_ult:,}")
    c2.metric("Total Acumulado 2021–2025", f"{TOTAL_ATENCIONES:,}")
    c3.metric("IPRESS Activas", f"{N_IPRESS:,}")
    c4.metric("Regiones", str(N_REGIONES))
    st.divider()

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        ts_plot = ts.copy()
        ts_plot["Semestre"] = ts_plot["PERIODO_NUM"].map(ETIQUETAS_PERIODO)
        ts_plot["Var%"] = ts_plot["ATENCIONES"].pct_change() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_plot["Semestre"], y=ts_plot["ATENCIONES"],
            mode="lines+markers+text",
            text=ts_plot["ATENCIONES"].apply(lambda x: f"{x/1e6:.1f}M"),
            textposition="top center",
            line=dict(color="#1976D2", width=3), marker=dict(size=8),
        ))
        fig.update_layout(title="Tendencia Nacional de Atenciones SIS",
                          height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        ts_serv = cargar_ts_servicio()
        serv_ult = ts_serv[ts_serv["PERIODO_NUM"] == ultimo_p]
        fig2 = px.pie(serv_ult, names="SERVICIO_CATEGORIA", values="ATENCIONES",
                      title=f"Composición por Servicio ({ETIQUETAS_PERIODO[ultimo_p]})",
                      color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig2.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Atenciones por Región")
    filtro_p = st.select_slider(
        "Seleccionar periodo",
        options=sorted(ts["PERIODO_NUM"].unique().tolist()),
        value=ultimo_p,
        format_func=lambda x: ETIQUETAS_PERIODO.get(x, str(x)),
    )
    top_reg = cargar_top_regiones()
    top_reg_f = (top_reg[top_reg["PERIODO_NUM"] == filtro_p]
                 .sort_values("ATENCIONES", ascending=False).head(15))
    fig3 = px.bar(top_reg_f, x="ATENCIONES", y="REGION", orientation="h",
                  color="ATENCIONES", color_continuous_scale="Blues",
                  title=f"Top 15 Regiones — {ETIQUETAS_PERIODO.get(filtro_p, filtro_p)}")
    fig3.update_layout(height=420, yaxis={"categoryorder":"total ascending"},
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        dist_n = cargar_dist_nivel()
        niv_tot = dist_n.groupby("NIVEL_EESS")["ATENCIONES"].sum().reset_index()
        fig4 = px.bar(niv_tot, x="NIVEL_EESS", y="ATENCIONES",
                      color="NIVEL_EESS",
                      color_discrete_sequence=["#1976D2","#42A5F5","#90CAF9"],
                      title="Atenciones por Nivel EESS")
        fig4.update_layout(showlegend=False, height=280, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        dist_e = cargar_dist_edad()
        fig5 = px.bar(dist_e, x="GRUPO_EDAD", y="ATENCIONES",
                      color="ATENCIONES", color_continuous_scale="Blues",
                      title="Atenciones por Grupo de Edad")
        fig5.update_layout(height=280, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig5, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# PÁGINA 2: PREDICCIÓN
# ═════════════════════════════════════════════════════════════════
elif pagina == "🔮 Predicción de Demanda":
    st.title("🔮 Predicción de Demanda de Atenciones")
    st.caption("Estimación para **2025 S2** basada en el modelo LightGBM (MAPE 53.96%).")

    cats    = cargar_catalogos()
    modelo  = cargar_modelo()
    enc_obj = cargar_encoder()
    clusters_df = cargar_clusters()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Establecimiento")
        region_sel = st.selectbox("Región", cats["regiones"])
        nivel_sel  = st.selectbox("Nivel EESS", cats["niveles"],
                                   help="I=Posta/Centro, II=Hospital apoyo, III=Hospital regional")
    with col2:
        st.subheader("Servicio")
        servicio_sel = st.selectbox("Tipo de Servicio", cats["servicios"])
    with col3:
        st.subheader("Perfil Demográfico")
        edad_sel = st.selectbox("Grupo de Edad", cats["edades"])
        sexo_sel = st.selectbox("Sexo", cats["sexos"])

    st.divider()
    st.subheader("Historial de Demanda")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        lag_input = st.number_input("Atenciones último semestre (2025 S1)",
                                     min_value=1, max_value=10000, value=10)
    with ch2:
        rolling_input = st.number_input("Promedio últimos 2 semestres",
                                         min_value=1, max_value=10000, value=lag_input)
    with ch3:
        tendencia_input = st.number_input("Tendencia (atenciones/semestre)",
                                           min_value=-500, max_value=500, value=0)
    n_periodos = st.slider("Semestres activos (madurez)", 1, 9, 5)

    cluster_default = -1
    if len(clusters_df) > 0 and "NIVEL" in clusters_df.columns:
        nivel_map = {"I":1,"II":2,"III":3}
        nm = nivel_map.get(nivel_sel, 1)
        mask = clusters_df["NIVEL"] == nm
        if mask.any():
            cluster_default = int(clusters_df[mask]["CLUSTER"].mode().iloc[0])

    st.divider()
    if st.button("🚀 Generar Predicción", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            pred = predecir(
                modelo=modelo, enc_obj=enc_obj,
                region=region_sel, nivel_eess=nivel_sel,
                desc_servicio=servicio_sel, grupo_edad=edad_sel,
                sexo=sexo_sel, semestre=2, periodo_num=10,
                lag_atenciones=lag_input, rolling_mean=rolling_input,
                tendencia=tendencia_input, n_periodos=n_periodos,
                cluster=cluster_default,
            )
        st.success(f"### Predicción: **{pred:,} atenciones** en 2025 S2")
        var = pred - lag_input
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric("Predicción 2025 S2", f"{pred:,}", f"{var:+,} vs último semestre")
        cr2.metric("Lag 2025 S1", f"{lag_input:,}")
        cr3.metric("Tendencia", f"{tendencia_input:+,}/sem")

        hist_p = list(range(max(1, 10 - n_periodos), 10))
        hist_v = [max(1, int(lag_input + tendencia_input * (i - len(hist_p))))
                  for i in range(len(hist_p))]
        hist_l = [ETIQUETAS_PERIODO.get(p, f"P{p}") for p in hist_p]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=hist_l, y=hist_v, mode="lines+markers",
                                   name="Historial", line=dict(color="#1976D2")))
        fig_p.add_trace(go.Scatter(x=["2025 S2"], y=[pred], mode="markers",
                                   name="Predicción",
                                   marker=dict(size=14, color="#E53935", symbol="star")))
        fig_p.update_layout(title="Proyección de Demanda", height=280,
                             margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig_p, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# PÁGINA 3: CLUSTERING
# ═════════════════════════════════════════════════════════════════
elif pagina == "🗂️ Segmentación de IPRESS":
    st.title("🗂️ Segmentación de Establecimientos de Salud")
    clusters_df = cargar_clusters()

    if clusters_df.empty:
        st.warning("No se encontró ipress_clusters.parquet.")
        st.stop()

    k = clusters_df["CLUSTER"].nunique()
    st.info(f"Se identificaron **{k} clusters** de IPRESS con perfiles diferenciados.")

    feats = [c for c in ["TOTAL_ATENCIONES","NIVEL","PCT_MATERNO","PCT_PREVENTIVO",
                          "PCT_TELEMATICA","PCT_FEMENINO","EDAD_MEDIA","N_PERIODOS"]
             if c in clusters_df.columns]
    perfil = clusters_df.groupby("CLUSTER")[feats].mean().round(2)
    perfil["n_ipress"] = clusters_df["CLUSTER"].value_counts().sort_index()
    st.subheader("Perfil promedio por cluster")
    st.dataframe(perfil.style.background_gradient(cmap="Blues", axis=0),
                 use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        tam = clusters_df["CLUSTER"].value_counts().reset_index()
        tam.columns = ["Cluster","N IPRESS"]
        tam["Cluster"] = "Cluster " + tam["Cluster"].astype(str)
        fig_t = px.bar(tam.sort_values("Cluster"), x="Cluster", y="N IPRESS",
                       color="N IPRESS", color_continuous_scale="Blues",
                       title="IPRESS por Cluster")
        fig_t.update_layout(showlegend=False, height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_t, use_container_width=True)

    with col_b:
        if "TOTAL_ATENCIONES" in clusters_df.columns:
            sc_data = clusters_df.copy()
            sc_data["Cluster"] = "Cluster " + sc_data["CLUSTER"].astype(str)
            y_col = "N_PERIODOS" if "N_PERIODOS" in sc_data.columns else "NIVEL"
            fig_s = px.scatter(
                sc_data.sample(min(2000, len(sc_data)), random_state=42),
                x="TOTAL_ATENCIONES", y=y_col, color="Cluster",
                size="TOTAL_ATENCIONES", size_max=20, opacity=0.7,
                title="Volumen vs Madurez del Historial")
            fig_s.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("Descripción de Clusters")
    descripciones = {
        0: "**Hospitales de referencia (Nivel II/III):** alto volumen, mix diverso, demanda estable.",
        1: "**Centros maternos-infantiles (Nivel I, sierra/selva):** alta proporción de servicios CRED y APN.",
        2: "**IPRESS con telemedicina:** alta adopción de teleatención en zonas remotas.",
        3: "**Postas nuevas / historial corto:** baja demanda, candidatas a cold-start.",
    }
    for cl in sorted(clusters_df["CLUSTER"].unique()):
        n = (clusters_df["CLUSTER"] == cl).sum()
        with st.expander(f"Cluster {cl} — {n:,} IPRESS"):
            st.markdown(descripciones.get(cl, "Perfil mixto."))

# ═════════════════════════════════════════════════════════════════
# PÁGINA 4: EVALUACIÓN
# ═════════════════════════════════════════════════════════════════
elif pagina == "📊 Evaluación del Modelo":
    st.title("📊 Evaluación del Modelo Predictivo")

    res_df = cargar_resultados_modelos()
    dist_err, nivel_err = cargar_eval_errores()
    pv_df  = cargar_propuesta_valor()

    if not res_df.empty:
        st.subheader("Comparativa de Modelos — Test (2024 S2)")
        test_res = res_df[res_df["conjunto"] == "Test"]
        mejor_idx = test_res["MAPE"].idxmin()
        cols_m = st.columns(len(test_res))
        for i, (_, row) in enumerate(test_res.iterrows()):
            cols_m[i].metric(
                row["modelo"].replace(" — Test",""),
                f"MAPE {row['MAPE']:.1f}%",
                delta=f"R² {row['R2']:.3f}",
            )
            if row.name == mejor_idx:
                cols_m[i].caption("★ Mejor modelo")
        st.divider()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_m = px.bar(res_df, x="modelo", y="MAPE", color="conjunto",
                           barmode="group", title="MAPE — Validación vs Test",
                           color_discrete_sequence=["#1976D2","#E53935"])
            fig_m.add_hline(y=55.89, line_dash="dash", line_color="orange",
                             annotation_text="Baseline ~55.9%")
            fig_m.update_layout(height=320, xaxis_tickangle=-25,
                                 margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_m, use_container_width=True)
        with col_g2:
            fig_r = px.bar(res_df, x="modelo", y="R2", color="conjunto",
                           barmode="group", title="R² — Validación vs Test",
                           color_discrete_sequence=["#1976D2","#E53935"])
            fig_r.add_hline(y=0.75, line_dash="dash", line_color="green",
                             annotation_text="Objetivo R²=0.75")
            fig_r.update_layout(height=320, xaxis_tickangle=-25,
                                 margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_r, use_container_width=True)

    if not nivel_err.empty:
        st.subheader("MAPE por Nivel EESS en Test")
        nivel_err["Nivel"] = nivel_err["NIVEL_NUM"].map(
            {1:"I (Posta)",2:"II (Centro)",3:"III (Hospital)"})
        fig_niv = px.bar(nivel_err, x="Nivel", y="ERROR_PCT",
                         color="ERROR_PCT", color_continuous_scale="Reds",
                         title="MAPE promedio por Nivel de Establecimiento")
        fig_niv.add_hline(y=53.96, line_dash="dash", line_color="blue",
                           annotation_text="MAPE global 53.96%")
        fig_niv.update_layout(showlegend=False, height=300,
                               margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_niv, use_container_width=True)

    if not pv_df.empty:
        st.subheader("Propuesta de Valor Cuantificada")
        col_pv1, col_pv2, col_pv3 = st.columns(3)
        rows = pv_df[pv_df["componente"] != "TOTAL ESTIMADO"]
        total = pv_df[pv_df["componente"] == "TOTAL ESTIMADO"]["ahorro_anual_soles"].values
        for i, (_, row) in enumerate(rows.iterrows()):
            [col_pv1, col_pv2][i].metric(
                row["componente"].replace(" por reducir subasignacion",""),
                f"S/. {row['ahorro_anual_soles']:,.0f}",
            )
        if len(total):
            col_pv3.metric("TOTAL AHORRO ANUAL ESTIMADO",
                           f"S/. {total[0]:,.0f}",
                           delta=f"Mejora relativa vs baseline: {pv_df['mejora_relativa_pct'].iloc[0]:.2f}%")

    st.info(
        "**Nota metodológica:** MAPE ~54% refleja la granularidad extremadamente fina "
        "(IPRESS × Servicio × Edad × Sexo) con mediana de 3 atenciones/registro. "
        "La autocorrelación temporal (r = 0.747) confirma que el modelo captura "
        "correctamente los patrones de demanda. Baseline real: 55.89% (NB06)."
    )

    # SHAP
    shap_path = Path(__file__).parent.parent / "reports" / "shap_importancias.csv"
    if shap_path.exists():
        st.subheader("Importancia de Features (SHAP — LightGBM)")
        shap_df = pd.read_csv(shap_path)
        fig_sh = px.bar(shap_df.head(12), x="mean_abs", y="feature", orientation="h",
                        color="mean_abs", color_continuous_scale="Blues",
                        title="Top 12 Features por SHAP",
                        labels={"mean_abs":"|SHAP| medio","feature":"Feature"})
        fig_sh.update_layout(yaxis={"categoryorder":"total ascending"},
                              height=380, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_sh, use_container_width=True)
