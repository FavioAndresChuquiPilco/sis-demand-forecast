import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    cargar_ts_nacional, cargar_ts_servicio,
    cargar_dist_nivel, cargar_dist_edad,
    cargar_top_regiones, cargar_catalogos, cargar_modelo,
    cargar_encoder, cargar_resultados_modelos,
    cargar_eval_errores, cargar_propuesta_valor,
    cargar_eda_univariado, cargar_eda_bivariado,
    cargar_eda_multivariado, cargar_correlaciones, cargar_shap,
    ETIQUETAS_PERIODO,
    TOTAL_ATENCIONES, N_IPRESS, N_REGIONES,
)
from utils.predictor import predecir

st.set_page_config(
    page_title="SIS Demand Forecast",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

NIVEL_LABEL = {
    "I": "Nivel I (Posta)", "II": "Nivel II (Centro)", "III": "Nivel III (Hospital)",
    1: "Nivel I (Posta)", 2: "Nivel II (Centro)", 3: "Nivel III (Hospital)",
}
EDAD_LABEL = {
    "00 - 04 ANOS": "0–4", "05 - 11 ANOS": "5–11", "12 - 17 ANOS": "12–17",
    "18 - 29 ANOS": "18–29", "30 - 59 ANOS": "30–59", "60 - MAS ANOS": "60+",
}
ORDEN_EDAD_CORTO = ["0–4", "5–11", "12–17", "18–29", "30–59", "60+"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 SIS-DEMAND")
    st.caption("Pronóstico de Demanda de Atenciones\nSeguro Integral de Salud — Perú")
    st.divider()
    pagina = st.selectbox(
        "Navegación",
        [
            "🏠 Dashboard Nacional",
            "🔍 Exploración de Datos",
            "🔮 Predicción de Demanda",
            "🧠 Interpretabilidad del Modelo",
            "📊 Evaluación del Modelo",
        ],
    )
    st.divider()
    st.caption("TAF — Analítica de Datos\nCENTRUM PUCP · 2026")

# =============================================================================
# PÁGINA 1: DASHBOARD NACIONAL
# =============================================================================
if pagina == "🏠 Dashboard Nacional":
    st.title("Dashboard Nacional — Atenciones SIS 2021–2025")

    ts = cargar_ts_nacional()
    ultimo_p  = int(ts["PERIODO_NUM"].max())
    total_ult = int(ts[ts["PERIODO_NUM"] == ultimo_p]["ATENCIONES"].sum())

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
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_plot["Semestre"], y=ts_plot["ATENCIONES"],
            mode="lines+markers+text",
            text=ts_plot["ATENCIONES"].apply(lambda x: f"{x/1e6:.1f}M"),
            textposition="top center",
            line=dict(color="#1976D2", width=3), marker=dict(size=8),
        ))
        fig.update_layout(title="Tendencia Nacional de Atenciones SIS",
                          height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        ts_serv = cargar_ts_servicio()
        serv_ult = ts_serv[ts_serv["PERIODO_NUM"] == ultimo_p]
        fig2 = px.pie(
            serv_ult, names="SERVICIO_CATEGORIA", values="ATENCIONES",
            title=f"Composición por Servicio ({ETIQUETAS_PERIODO[ultimo_p]})",
            color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4,
        )
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Atenciones por Región")
    filtro_p = st.select_slider(
        "Seleccionar periodo",
        options=sorted(ts["PERIODO_NUM"].unique().tolist()),
        value=ultimo_p,
        format_func=lambda x: ETIQUETAS_PERIODO.get(x, str(x)),
    )
    top_reg = cargar_top_regiones()
    top_reg_f = (
        top_reg[top_reg["PERIODO_NUM"] == filtro_p]
        .sort_values("ATENCIONES", ascending=False)
        .head(15)
    )
    fig3 = px.bar(
        top_reg_f, x="ATENCIONES", y="REGION", orientation="h",
        color="ATENCIONES", color_continuous_scale="Blues",
        title=f"Top 15 Regiones — {ETIQUETAS_PERIODO.get(filtro_p, filtro_p)}",
    )
    fig3.update_layout(height=420, yaxis={"categoryorder": "total ascending"},
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        dist_n = cargar_dist_nivel()
        niv_tot = dist_n.groupby("NIVEL_EESS")["ATENCIONES"].sum().reset_index()
        niv_tot["Nivel"] = niv_tot["NIVEL_EESS"].map(NIVEL_LABEL)
        niv_tot = (
            niv_tot[niv_tot["Nivel"].notna()]
            .sort_values("Nivel")
        )
        fig4 = px.bar(
            niv_tot, x="Nivel", y="ATENCIONES",
            color="Nivel",
            color_discrete_sequence=["#1976D2", "#42A5F5", "#90CAF9"],
            title="Atenciones por Nivel EESS",
            text=niv_tot["ATENCIONES"].apply(lambda x: f"{x/1e6:.1f}M"),
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, height=300,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        dist_e = cargar_dist_edad()
        dist_e = dist_e.copy()
        dist_e["Edad"] = dist_e["GRUPO_EDAD"].map(EDAD_LABEL).fillna(dist_e["GRUPO_EDAD"])
        dist_e["Edad"] = pd.Categorical(dist_e["Edad"], categories=ORDEN_EDAD_CORTO, ordered=True)
        dist_e = dist_e.sort_values("Edad")
        fig5 = px.bar(
            dist_e, x="Edad", y="ATENCIONES",
            color="ATENCIONES", color_continuous_scale="Blues",
            title="Atenciones por Grupo de Edad",
            text=dist_e["ATENCIONES"].apply(lambda x: f"{x/1e6:.1f}M"),
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(height=300, showlegend=False,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig5, use_container_width=True)

# =============================================================================
# PÁGINA 2: EXPLORACIÓN DE DATOS (EDA)
# =============================================================================
elif pagina == "🔍 Exploración de Datos":
    st.title("🔍 Exploración de Datos — Hallazgos EDA")
    st.caption(
        "Análisis exploratorio sobre **41.9 millones de registros** SIS 2021–2025 "
        "(9 semestres, 8,725 IPRESS, 26 regiones, 65 servicios)."
    )

    VAR_LABELS = {
        "REGION": "Región", "NIVEL_EESS": "Nivel EESS", "PLAN_SEGURO": "Plan Seguro",
        "SEXO": "Sexo", "GRUPO_EDAD": "Grupo de Edad",
        "DESC_SERVICIO": "Tipo de Servicio", "SEMESTRE": "Semestre",
    }

    # ── Univariado ────────────────────────────────────────────────────────────
    st.subheader("Análisis Univariado — Distribución de Variables")
    st.markdown("Categoría con mayor concentración por variable (% sobre total de atenciones).")

    univ = cargar_eda_univariado()
    if not univ.empty:
        univ["Variable"] = univ["variable"].map(VAR_LABELS).fillna(univ["variable"])
        univ_sorted = univ.sort_values("top1_pct")

        col_u1, col_u2 = st.columns([3, 2])
        with col_u1:
            fig_univ = px.bar(
                univ_sorted, x="top1_pct", y="Variable", orientation="h",
                text=univ_sorted.apply(
                    lambda r: f"{r['top1_valor']} ({r['top1_pct']:.1f}%)", axis=1
                ),
                color="top1_pct", color_continuous_scale="Blues",
                title="Categoría dominante por variable (%)",
                labels={"top1_pct": "% del total", "Variable": ""},
            )
            fig_univ.update_traces(textposition="outside")
            fig_univ.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0),
                                   showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_univ, use_container_width=True)

        with col_u2:
            st.markdown("**Hallazgos clave:**")
            st.markdown("""
**Demanda altamente concentrada en el primer nivel:**
- **94%** de atenciones en establecimientos de **Nivel I** (postas y centros de salud)
- Solo **6%** en hospitales de apoyo y referencia

**Perfil del asegurado SIS:**
- **74%** bajo plan SIS Gratuito (población en pobreza)
- **60%** de atenciones son de **mujeres** (salud materna-infantil)
- El grupo de **30–59 años** concentra la mayor carga (21.9%)

**Distribución geográfica:**
- **CAJAMARCA** lidera con 9.8% del total nacional
- Alta heterogeneidad entre las 26 regiones

**Estacionalidad:**
- **Semestre 1** (Ene–Jun) concentra el **55.6%** de atenciones anuales
            """)

    # ── Bivariado ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Análisis Bivariado — Significancia Estadística")
    st.markdown(
        "Pruebas no paramétricas para verificar si cada variable discrimina "
        "significativamente el nivel de demanda."
    )

    biv = cargar_eda_bivariado()
    if not biv.empty:
        col_b1, col_b2 = st.columns([2, 3])
        with col_b1:
            biv_disp = biv.copy()
            biv_disp["Variable"] = biv_disp["variable"].map(VAR_LABELS).fillna(biv_disp["variable"])
            biv_disp["p-valor"] = biv_disp["p_value"].apply(
                lambda x: "< 1e-300" if x == 0 else f"{x:.2e}"
            )
            biv_disp["Sig."] = biv_disp["interpretacion"]
            st.dataframe(
                biv_disp[["Variable", "test", "p-valor", "Sig."]].set_index("Variable"),
                use_container_width=True,
            )
        with col_b2:
            st.success(
                "**Todas las variables son estadísticamente significativas** (p ≈ 0, nivel \*\*\*).\n\n"
                "Esto confirma que ninguna variable es redundante en el modelo."
            )
            st.markdown("""
| Variable | Test | Por qué discrimina |
|---|---|---|
| **Grupo de Edad** | Kruskal-Wallis | Estadístico más alto (657,544) — mayor poder discriminante |
| **Región** | Kruskal-Wallis | Alta heterogeneidad geográfica en la demanda (268,561) |
| **Nivel EESS** | Kruskal-Wallis | Postas vs hospitales difieren radicalmente en carga |
| **Sexo** | Mann-Whitney | Mujeres concentran más atenciones preventivas y maternas |
| **Semestre** | Mann-Whitney | Estacionalidad confirmada — S1 > S2 sistemáticamente |
            """)

    # ── Correlaciones ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Correlación con el Objetivo (log Atenciones)")

    corr = cargar_correlaciones()
    if not corr.empty:
        corr.columns = ["feature", "correlacion"]
        corr = corr[corr["feature"] != "ATENCIONES"].copy()
        FEAT_LABELS = {
            "NIVEL_EESS_ENC": "Nivel EESS",
            "SEXO_ENC": "Sexo",
            "DESC_SERVICIO_ENC": "Tipo de Servicio",
            "GRUPO_EDAD_ENC": "Grupo de Edad",
            "REGION_ENC": "Región",
            "PERIODO_NUM": "Periodo (tiempo)",
            "SEMESTRE": "Semestre",
        }
        corr["Feature"] = corr["feature"].map(FEAT_LABELS).fillna(corr["feature"])
        corr["Correlación"] = corr["correlacion"].astype(float)
        corr["Tipo"] = corr["Correlación"].apply(lambda x: "Positiva" if x >= 0 else "Negativa")
        corr = corr.sort_values("Correlación")

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_corr = px.bar(
                corr, x="Correlación", y="Feature", orientation="h",
                color="Tipo",
                color_discrete_map={"Positiva": "#1976D2", "Negativa": "#E53935"},
                title="Correlación de Pearson con log(Atenciones)",
                text=corr["Correlación"].apply(lambda x: f"{x:+.3f}"),
            )
            fig_corr.update_traces(textposition="outside")
            fig_corr.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                   showlegend=False)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_c2:
            st.info(
                "Las correlaciones lineales son bajas (~0.09 máx.), lo que **no** significa "
                "que las variables sean irrelevantes. Significa que la relación es **no lineal** "
                "— exactamente por eso LightGBM (árboles) supera a la regresión lineal.\n\n"
                "- **Nivel EESS** (r=+0.09): hospitales de mayor nivel → más atenciones por registro\n"
                "- **Sexo** (r=−0.07): mujeres → registros con ligeramente más atenciones"
            )

    # ── Multivariado ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Análisis Multivariado — PCA, Multicolinealidad y RF Exploratorio")

    multi = cargar_eda_multivariado()
    if not multi.empty:

        def get_val(df, analisis, metrica):
            row = df[(df["analisis"] == analisis) & (df["metrica"] == metrica)]["valor"]
            return row.values[0] if len(row) else None

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown("#### PCA — Dimensionalidad")
            n_comp = get_val(multi, "PCA", "n_componentes_90pct_varianza")
            var1   = get_val(multi, "PCA", "varianza_PC1")
            var2   = get_val(multi, "PCA", "varianza_PC2")
            if n_comp:
                st.metric("Componentes para 90% varianza", int(float(n_comp)))
            if var1:
                st.metric("Varianza explicada por PC1", f"{float(var1):.1%}")
            if var2:
                st.metric("Varianza explicada por PC2", f"{float(var2):.1%}")
            st.caption(
                "Se requieren **6 componentes** para capturar el 90% de la varianza. "
                "El espacio es moderadamente complejo — baja redundancia entre variables."
            )

        with col_m2:
            st.markdown("#### VIF — Multicolinealidad")
            max_vif  = get_val(multi, "VIF", "max_VIF")
            feat_vif = get_val(multi, "VIF", "feature_max_VIF")
            sin_mc   = get_val(multi, "VIF", "sin_multicolinealidad_severa")
            if max_vif:
                st.metric("VIF máximo entre features", f"{float(max_vif):.3f}")
            if feat_vif:
                st.metric("Feature con mayor VIF", str(feat_vif))
            st.metric("¿Multicolinealidad severa?", "No" if str(sin_mc) == "True" else "Revisar")
            st.caption(
                "**VIF < 5 → sin multicolinealidad**. Las features son independientes entre sí. "
                "No se requiere eliminar ninguna variable por colinealidad."
            )

        with col_m3:
            st.markdown("#### RF Exploratorio")
            r2_rf    = get_val(multi, "RF_preliminar", "R2_muestra")
            top_feat = get_val(multi, "RF_preliminar", "top_feature")
            if r2_rf:
                st.metric("R² RF en muestra", f"{float(r2_rf):.3f}")
            if top_feat:
                st.metric("Feature más importante", str(top_feat))
            st.caption(
                "**R²=0.598 en muestra** confirma que un modelo de árboles puede capturar "
                "los patrones de demanda. El feature más importante es el tipo de servicio "
                "(DESC_SERVICIO_ENC)."
            )

        # Cold start
        pct_cold  = get_val(multi, "cold_start", "pct_cold_start")
        keys_nvas = get_val(multi, "cold_start", "keys_nuevas_P8_P9")
        tot_keys  = get_val(multi, "cold_start", "total_keys_unicas")
        if pct_cold:
            st.divider()
            st.markdown("#### Cold Start — IPRESS sin historial previo")
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Combinaciones IPRESS únicas", f"{int(float(tot_keys)):,}" if tot_keys else "—")
            cs2.metric("Nuevas en periodos recientes", f"{int(float(keys_nvas)):,}" if keys_nvas else "—")
            cs3.metric("% Cold Start", f"{float(pct_cold):.2f}%")
            st.info(
                "El **5.83%** de las IPRESS en periodos recientes no tiene historial previo. "
                "Se imputan con la mediana de establecimientos similares "
                "(mismo Nivel + Región + Servicio). Este tratamiento evita sesgos en el modelo."
            )

# =============================================================================
# PÁGINA 3: PREDICCIÓN
# =============================================================================
elif pagina == "🔮 Predicción de Demanda":
    st.title("🔮 Predicción de Demanda de Atenciones")
    st.caption("Estimación para **2025 S2** basada en el modelo LightGBM (MAPE 53.96%).")

    cats    = cargar_catalogos()
    modelo  = cargar_modelo()
    enc_obj = cargar_encoder()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Establecimiento")
        region_sel = st.selectbox("Región", cats["regiones"])
        nivel_sel  = st.selectbox(
            "Nivel EESS", cats["niveles"],
            help="I = Posta/Centro de salud | II = Hospital de apoyo | III = Hospital regional",
        )
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
        lag_input = st.number_input(
            "Atenciones último semestre (2025 S1)", min_value=1, max_value=10_000, value=10,
        )
    with ch2:
        rolling_input = st.number_input(
            "Promedio últimos 2 semestres", min_value=1, max_value=10_000, value=lag_input,
        )
    with ch3:
        tendencia_input = st.number_input(
            "Tendencia (atenciones/semestre)", min_value=-500, max_value=500, value=0,
        )
    n_periodos = st.slider("Semestres activos (madurez del establecimiento)", 1, 9, 5)

    nivel_num_map = {"I": 1, "II": 2, "III": 3}
    cluster_default = nivel_num_map.get(nivel_sel, 1) - 1

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
        cr2.metric("Lag 2025 S1 (input)", f"{lag_input:,}")
        cr3.metric("Tendencia ingresada", f"{tendencia_input:+,}/sem")

        hist_p = list(range(max(1, 10 - n_periodos), 10))
        hist_v = [max(1, int(lag_input + tendencia_input * (i - len(hist_p))))
                  for i in range(len(hist_p))]
        hist_l = [ETIQUETAS_PERIODO.get(p, f"P{p}") for p in hist_p]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=hist_l, y=hist_v, mode="lines+markers",
            name="Historial estimado", line=dict(color="#1976D2"),
        ))
        fig_p.add_trace(go.Scatter(
            x=["2025 S2"], y=[pred], mode="markers",
            name="Predicción LightGBM",
            marker=dict(size=14, color="#E53935", symbol="star"),
        ))
        fig_p.update_layout(
            title="Proyección de Demanda", height=300,
            margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.info(
            "**Nota:** La predicción usa los mismos features que el modelo de producción: "
            "promedio móvil, tendencia lineal, periodos activos, nivel, región y servicio. "
            "MAPE promedio en test: **53.96%** (nivel de detalle muy fino: IPRESS × servicio × edad × sexo)."
        )

# =============================================================================
# PÁGINA 4: INTERPRETABILIDAD
# =============================================================================
elif pagina == "🧠 Interpretabilidad del Modelo":
    st.title("🧠 Interpretabilidad del Modelo — Análisis SHAP")
    st.caption(
        "SHAP (SHapley Additive exPlanations) mide la **contribución real de cada feature** "
        "a la predicción. A diferencia de la importancia de Gini, SHAP es teóricamente sólido "
        "y permite explicar predicciones individuales."
    )

    shap_df = cargar_shap()

    if shap_df.empty:
        st.warning("No se encontró shap_importancias.csv en reports/.")
        st.stop()

    shap_df = shap_df.sort_values("mean_abs", ascending=False).reset_index(drop=True)

    # ── Gráfico principal SHAP ────────────────────────────────────────────────
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        SHAP_LABELS = {
            "ROLLING_MEAN_2SEM": "Promedio móvil 2 semestres",
            "TENDENCIA_LINEAL": "Tendencia lineal (OLS)",
            "N_PERIODOS_ACTIVOS": "Periodos activos (madurez)",
            "GRUPO_EDAD_ORD": "Grupo de edad",
            "DESC_SERVICIO": "Tipo de servicio",
            "PERIODO_NUM": "Número de periodo (tiempo)",
            "SEXO_BIN": "Sexo (binario)",
            "REGION": "Región",
            "LAG_ATENCIONES_1SEM": "Lag 1 semestre",
            "SERVICIO_CATEGORIA": "Categoría servicio",
            "ES_MATERNO": "¿Servicio materno?",
            "REGION_TIPO": "Tipo de región",
            "NIVEL_NUM": "Nivel EESS",
            "SEMESTRE": "Semestre (1 o 2)",
            "CLUSTER": "Cluster IPRESS",
            "ES_PREVENTIVO": "¿Servicio preventivo?",
            "ES_TELEMATICA": "¿Telemedicina?",
        }
        shap_plot = shap_df.head(12).copy()
        shap_plot["Feature Label"] = shap_plot["feature"].map(SHAP_LABELS).fillna(shap_plot["feature"])
        shap_plot = shap_plot.sort_values("mean_abs")

        fig_sh = px.bar(
            shap_plot, x="mean_abs", y="Feature Label", orientation="h",
            color="mean_abs", color_continuous_scale="Blues",
            title="Top 12 Features — Importancia SHAP (|SHAP| medio)",
            labels={"mean_abs": "|SHAP| medio", "Feature Label": ""},
            text=shap_plot["mean_abs"].apply(lambda x: f"{x:.4f}"),
        )
        fig_sh.update_traces(textposition="outside")
        fig_sh.update_layout(
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    with col_s2:
        st.markdown("### ¿Qué significa cada feature?")
        st.markdown("""
**1. Promedio móvil 2 semestres** (`ROLLING_MEAN_2SEM`)
El predictor más poderoso (SHAP=0.351). Captura la demanda reciente promedio — si un IPRESS atendía 50 casos/semestre, el modelo espera ~50 en el siguiente.

**2. Tendencia lineal** (`TENDENCIA_LINEAL`)
La pendiente OLS de la serie histórica de cada IPRESS. Si la demanda crece 5 casos/semestre, el modelo lo incorpora directamente.

**3. Periodos activos** (`N_PERIODOS_ACTIVOS`)
Cuántos semestres lleva activo el IPRESS. Establecimientos nuevos (cold start) tienen demanda más impredecible.

**4. Grupo de edad** y **Tipo de servicio**
Definen el perfil de la demanda: pediátrico vs adulto mayor, preventivo vs curativo.

**5. Número de periodo**
Captura la tendencia secular nacional (el SIS crece con el tiempo).
        """)

    # ── Tabla completa ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tabla Completa de Importancias SHAP")
    shap_full = shap_df.copy()
    shap_full["Feature (nombre técnico)"] = shap_full["feature"]
    shap_full["Feature (descripción)"] = shap_full["feature"].map(SHAP_LABELS).fillna(shap_full["feature"])
    shap_full["|SHAP| medio"] = shap_full["mean_abs"].round(4)
    shap_full["% contribución"] = (shap_full["mean_abs"] / shap_full["mean_abs"].sum() * 100).round(1)
    st.dataframe(
        shap_full[["Feature (nombre técnico)", "Feature (descripción)", "|SHAP| medio", "% contribución"]],
        use_container_width=True, hide_index=True,
    )

    # ── Conclusiones ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Conclusiones de Interpretabilidad")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.success("""
**El modelo es interpretable y sensato:**

- Los 3 features más importantes son el **historial reciente** del IPRESS
  (`ROLLING_MEAN`, `TENDENCIA`, `N_PERIODOS`) — explican el **54%** de la importancia total
- El modelo **no memoriza** valores únicos: usa patrones temporales y de perfil
- Las variables demográficas (`GRUPO_EDAD`, `SEXO`) y de servicio aportan exactamente
  la relevancia esperada según el conocimiento del dominio SIS
        """)
    with col_i2:
        st.info("""
**¿Por qué importa esto para el negocio?**

Un gestor de MINSA puede confiar en el modelo porque:

1. Si el IPRESS X atendió ~30 casos/semestre históricamente → predice ~30 (no algo arbitrario)
2. Si la tendencia es creciente → el modelo lo refleja en la proyección
3. Las variables de sexo, edad y servicio permiten **segmentar la demanda predicha** por perfil demográfico para planificar insumos y personal
        """)

# =============================================================================
# PÁGINA 5: EVALUACIÓN DEL MODELO
# =============================================================================
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
        for i, (idx, row) in enumerate(test_res.iterrows()):
            cols_m[i].metric(
                row["modelo"].replace(" — Test", "").replace(" -- Test", ""),
                f"MAPE {row['MAPE']:.1f}%",
                delta=f"R² {row['R2']:.3f}",
            )
            if idx == mejor_idx:
                cols_m[i].caption("★ Mejor modelo")
        st.divider()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_m = px.bar(
                res_df, x="modelo", y="MAPE", color="conjunto",
                barmode="group", title="MAPE — Validación vs Test",
                color_discrete_sequence=["#1976D2", "#E53935"],
            )
            fig_m.add_hline(y=55.89, line_dash="dash", line_color="orange",
                            annotation_text="Baseline ~55.9%")
            fig_m.update_layout(height=320, xaxis_tickangle=-25,
                                 margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_m, use_container_width=True)
        with col_g2:
            fig_r = px.bar(
                res_df, x="modelo", y="R2", color="conjunto",
                barmode="group", title="R² — Validación vs Test",
                color_discrete_sequence=["#1976D2", "#E53935"],
            )
            fig_r.add_hline(y=0.75, line_dash="dash", line_color="green",
                            annotation_text="Objetivo R²=0.75")
            fig_r.update_layout(height=320, xaxis_tickangle=-25,
                                 margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_r, use_container_width=True)

    if not nivel_err.empty:
        st.subheader("MAPE por Nivel EESS en Test (2024 S2)")
        nivel_err = nivel_err.copy()
        nivel_err["Nivel"] = nivel_err["NIVEL_NUM"].map(
            {1: "Nivel I (Posta)", 2: "Nivel II (Centro)", 3: "Nivel III (Hospital)"}
        )
        fig_niv = px.bar(
            nivel_err[nivel_err["Nivel"].notna()],
            x="Nivel", y="ERROR_PCT",
            color="ERROR_PCT", color_continuous_scale="Reds",
            title="MAPE promedio por Nivel de Establecimiento",
            text=nivel_err[nivel_err["Nivel"].notna()]["ERROR_PCT"].apply(
                lambda x: f"{x:.1f}%"
            ),
        )
        fig_niv.update_traces(textposition="outside")
        fig_niv.add_hline(y=53.96, line_dash="dash", line_color="blue",
                           annotation_text="MAPE global 53.96%")
        fig_niv.update_layout(showlegend=False, height=320,
                               margin=dict(l=0, r=0, t=40, b=0),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_niv, use_container_width=True)

    if not pv_df.empty:
        st.subheader("Propuesta de Valor Cuantificada")
        rows  = pv_df[pv_df["componente"] != "TOTAL ESTIMADO"]
        total = pv_df[pv_df["componente"] == "TOTAL ESTIMADO"]["ahorro_anual_soles"].values
        col_pv1, col_pv2, col_pv3 = st.columns(3)
        for i, (_, row) in enumerate(rows.iterrows()):
            [col_pv1, col_pv2][i % 2].metric(
                row["componente"].replace(" por reducir subasignacion", ""),
                f"S/. {row['ahorro_anual_soles']:,.0f}",
            )
        if len(total):
            col_pv3.metric(
                "TOTAL AHORRO ANUAL ESTIMADO",
                f"S/. {total[0]:,.0f}",
                delta=f"Mejora vs baseline: {pv_df['mejora_relativa_pct'].iloc[0]:.2f}%",
            )

    st.info(
        "**Nota metodológica:** MAPE ~54% refleja la granularidad extremadamente fina "
        "(IPRESS × Servicio × Edad × Sexo) con mediana de 3 atenciones/registro. "
        "La autocorrelación temporal (r=0.747) confirma que el modelo captura correctamente "
        "los patrones de demanda. Baseline real: **55.89%** (NB06)."
    )
