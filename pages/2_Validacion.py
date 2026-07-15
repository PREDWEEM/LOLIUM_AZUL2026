# -*- coding: utf-8 -*-
"""Página Streamlit de validación automática para PREDWEEM Azul."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import calcular_metricas_validacion as motor

st.set_page_config(
    page_title=f"Validación — {motor.SITIO}",
    page_icon="📊",
    layout="wide",
)

ARCHIVOS_REQUERIDOS = (motor.APP, motor.METEO, motor.CAMPO, *motor.ARCHIVOS_MODELO)


def formato_fecha(valor: Any) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    return pd.Timestamp(valor).strftime("%d-%m-%Y")


def formato_numero(valor: Any, decimales: int = 2, sufijo: str = "") -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    return f"{float(valor):.{decimales}f}{sufijo}"


def huella_archivos(base: Path) -> tuple[tuple[str, int, int], ...]:
    huella: list[tuple[str, int, int]] = []
    for nombre in ARCHIVOS_REQUERIDOS:
        ruta = base / nombre
        if ruta.exists():
            estado = ruta.stat()
            huella.append((nombre, int(estado.st_mtime_ns), int(estado.st_size)))
        else:
            huella.append((nombre, -1, -1))
    return tuple(huella)


def ejecutar_sin_interferir_streamlit(base: Path, campo_acumulado: bool) -> dict[str, Any]:
    streamlit_real = sys.modules.get("streamlit")
    try:
        return motor.ejecutar_app(base, campo_acumulado=campo_acumulado)
    finally:
        if streamlit_real is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = streamlit_real


@st.cache_data(show_spinner=False)
def calcular_resultados(
    base_str: str,
    campo_acumulado: bool,
    umbral_evento: float,
    prominencia_pico: float,
    _huella: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base = Path(base_str)
    motor.UMBRAL_EVENTO = float(umbral_evento)
    motor.PROMINENCIA_PICO = float(prominencia_pico)

    globales = ejecutar_sin_interferir_streamlit(base, campo_acumulado)
    sincronizado = motor.sincronizar_eventos(
        globales["df"],
        globales["df_campo"],
        globales["col_fecha"],
        globales["col_plm2"],
    )
    indicadores = motor.metricas(globales, sincronizado)
    columnas = [
        c for c in (
            "Fecha",
            "EMERREL",
            "DG",
            "Primer_Pico_Habilitado",
            "EMERREL_ANTES_FILTRO_PRIMER_PICO",
        ) if c in globales["df"].columns
    ]
    diario = globales["df"][columnas].copy()
    return indicadores, sincronizado, diario


def grafico_flujos(sync: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sync["Fecha"], y=sync["Campo_Relativo"], name="Observado",
        marker_color="#60A5FA",
        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Observado: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=sync["Fecha"], y=sync["Sim_Relativo"], name="Simulado",
        marker_color="#166534",
        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Simulado: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Flujos relativos por intervalo real de monitoreo",
        barmode="group", xaxis_title="Fecha final del intervalo",
        yaxis_title="Flujo relativo", height=500, hovermode="x unified",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=25, t=80, b=65),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)")
    fig.update_xaxes(showgrid=False)
    return fig


def grafico_acumulado(sync: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sync["Fecha"], y=sync["Campo_Acumulado"] * 100,
        mode="markers+lines", name="Observado",
        marker=dict(size=9, color="#60A5FA", line=dict(color="#FFFFFF", width=1)),
        line=dict(color="#60A5FA", width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=sync["Fecha"], y=sync["Sim_Acumulado"] * 100,
        mode="lines", name="Simulado",
        line=dict(color="#166534", width=2.8, dash="dash"),
    ))
    fig.update_layout(
        title="Trayectoria acumulada observada y simulada",
        xaxis_title="Fecha", yaxis_title="Emergencia acumulada (%)",
        height=500, hovermode="x unified",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=25, t=80, b=65),
    )
    fig.update_yaxes(range=[0, 105], gridcolor="rgba(148,163,184,0.25)")
    fig.update_xaxes(showgrid=False)
    return fig


def grafico_decision(diario: pd.DataFrame, indicadores: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diario["Fecha"], y=diario["EMERREL"], mode="lines",
        name="EMERREL diaria", line=dict(color="#075FCF", width=2.3),
        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>EMERREL: %{y:.3f}<extra></extra>",
    ))

    control = indicadores.get("Fecha_control")
    limite = indicadores.get("Fecha_limite")
    inicio = indicadores.get("Fecha_inicio_termico")
    alerta = indicadores.get("Fecha_alerta")

    if pd.notna(control) and pd.notna(limite):
        fig.add_vrect(
            x0=control, x1=limite, fillcolor="rgba(34,197,94,0.12)",
            layer="below", line_width=0,
            annotation_text="Ventana eficiente", annotation_position="top left",
        )

    for fecha, texto, color, estilo in (
        (alerta, "Primera alerta", "#6B7280", "dash"),
        (inicio, "Inicio térmico", "#111827", "dot"),
        (control, "Control", "#111827", "dot"),
        (limite, "Límite", "#166534", "dot"),
    ):
        if pd.isna(fecha):
            continue
        fig.add_vline(x=fecha, line_color=color, line_dash=estilo, line_width=1.5)
        fig.add_annotation(
            x=fecha, xref="x", y=1.02, yref="paper", text=texto,
            showarrow=False, xanchor="center", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.93)", bordercolor="rgba(148,163,184,0.45)",
            borderwidth=1, borderpad=3, font=dict(size=11, color=color),
        )

    fig.update_layout(
        title="Serie diaria y fechas de decisión agronómica",
        xaxis_title="Fecha", yaxis_title="EMERREL", height=520,
        hovermode="x unified", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=70, r=25, t=105, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 1.05], gridcolor="rgba(148,163,184,0.25)")
    fig.update_xaxes(showgrid=False)
    return fig


def crear_excel(indicadores: dict[str, Any], sync: pd.DataFrame, diario: pd.DataFrame) -> bytes:
    salida = io.BytesIO()
    limpio = {k: motor.serializar(v) for k, v in indicadores.items()}
    with pd.ExcelWriter(salida, engine="xlsxwriter", datetime_format="dd-mm-yyyy") as writer:
        pd.DataFrame([limpio]).to_excel(writer, sheet_name="Metricas", index=False)
        sync.to_excel(writer, sheet_name="Event_to_Event", index=False)
        diario.to_excel(writer, sheet_name="Serie_Diaria", index=False)
    return salida.getvalue()


st.title(f"📊 Validación automática — {motor.SITIO}")
st.caption(
    "Comparación Event-to-Event entre la emergencia diaria simulada y "
    "los conteos reales almacenados en el repositorio."
)

faltantes = [nombre for nombre in ARCHIVOS_REQUERIDOS if not (BASE / nombre).exists()]

with st.sidebar:
    st.header("Configuración de validación")
    campo_acumulado = st.checkbox(
        "Los datos de campo son acumulados",
        value=bool(motor.CAMPO_ES_ACUMULADO),
        help="Activar solo cuando cada conteo sea el total acumulado desde el inicio.",
    )
    umbral_evento = st.number_input(
        "Umbral de flujo significativo", 0.00, 1.00,
        value=float(motor.UMBRAL_EVENTO), step=0.01, format="%.2f",
    )
    prominencia_pico = st.number_input(
        "Prominencia mínima del pico", 0.00, 1.00,
        value=float(motor.PROMINENCIA_PICO), step=0.01, format="%.2f",
    )
    recalcular = st.button(
        "🔄 Recalcular métricas", type="primary", width="stretch",
        disabled=bool(faltantes),
    )
    st.divider()
    st.code(
        f"Modelo: {motor.APP}\nMeteorología: {motor.METEO}\nCampo: {motor.CAMPO}",
        language=None,
    )

if faltantes:
    st.error("Faltan archivos en el repositorio:\n\n- " + "\n- ".join(faltantes))
    st.stop()

if recalcular:
    calcular_resultados.clear()

with st.spinner("Ejecutando PREDWEEM Azul y calculando métricas..."):
    try:
        indicadores, sincronizado, diario = calcular_resultados(
            str(BASE), campo_acumulado, float(umbral_evento),
            float(prominencia_pico), huella_archivos(BASE),
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.subheader("Resumen de desempeño")
resumen = st.columns(6)
resumen[0].metric("F1 de picos", formato_numero(indicadores.get("F1_picos"), 2))
resumen[1].metric("NSE de flujos", formato_numero(indicadores.get("NSE_flujos"), 2))
resumen[2].metric(
    "Δ primer pico", formato_numero(indicadores.get("Delta_primer_pico_dias"), 0, " d"),
    help="Negativo: anticipación. Positivo: retraso del modelo.",
)
resumen[3].metric("PEC al control", formato_numero(indicadores.get("PEC_control_pct"), 1, " %"))
resumen[4].metric("Lead time", formato_numero(indicadores.get("Lead_time_dias"), 0, " d"))
resumen[5].metric("Ventana 600–800", formato_numero(indicadores.get("Ventana_600_800_dias"), 0, " d"))

picos = st.columns(4)
picos[0].metric("Picos observados", formato_numero(indicadores.get("Picos_observados"), 0))
picos[1].metric("Picos simulados", formato_numero(indicadores.get("Picos_simulados"), 0))
picos[2].metric("Picos coincidentes", formato_numero(indicadores.get("Hits_picos"), 0))
picos[3].metric("Falsos picos", formato_numero(indicadores.get("Falsos_picos"), 0))

st.subheader("Fechas de decisión agronómica")
fechas = st.columns(4)
fechas[0].metric("Primer pico observado", formato_fecha(indicadores.get("Fecha_primer_pico_observado")))
fechas[1].metric("Inicio térmico", formato_fecha(indicadores.get("Fecha_inicio_termico")))
fechas[2].metric("Control recomendado", formato_fecha(indicadores.get("Fecha_control")))
fechas[3].metric("Límite de control", formato_fecha(indicadores.get("Fecha_limite")))

tab_flujos, tab_acum, tab_control, tab_descarga = st.tabs(
    ["Picos y flujos", "Trayectoria acumulada", "Decisión de control", "Datos y descargas"]
)

config_png = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": 1800, "height": 1000, "scale": 2},
}

with tab_flujos:
    st.plotly_chart(grafico_flujos(sincronizado), width="stretch", config=config_png)
    st.dataframe(sincronizado, width="stretch", hide_index=True)

with tab_acum:
    st.plotly_chart(grafico_acumulado(sincronizado), width="stretch", config=config_png)

with tab_control:
    st.plotly_chart(grafico_decision(diario, indicadores), width="stretch", config=config_png)
    st.info("La franja verde representa la ventana térmica entre 600 y 800 °Cd.")

with tab_descarga:
    limpio = {k: motor.serializar(v) for k, v in indicadores.items()}
    st.dataframe(pd.DataFrame([limpio]), width="stretch", hide_index=True)

    excel = crear_excel(indicadores, sincronizado, diario)
    csv_metricas = pd.DataFrame([limpio]).to_csv(index=False).encode("utf-8-sig")
    csv_eventos = sincronizado.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")
    json_metricas = json.dumps(limpio, ensure_ascii=False, indent=2).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "📥 Descargar Excel", excel, "PREDWEEM_validacion_azul.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
    )
    d2.download_button(
        "📥 Métricas CSV", csv_metricas, "metricas_azul.csv", "text/csv", width="stretch",
    )
    d3.download_button(
        "📥 Métricas JSON", json_metricas, "metricas_azul.json", "application/json", width="stretch",
    )
    st.download_button(
        "📥 Event-to-Event CSV", csv_eventos, "event_to_event_azul.csv",
        "text/csv", width="stretch",
    )

with st.expander("Definición de indicadores"):
    st.markdown(
        """
- **F1 de picos:** coincidencia entre máximos locales observados y simulados.
- **NSE de flujos:** ajuste de las magnitudes relativas Event-to-Event.
- **Δ primer pico:** fecha simulada menos fecha observada.
- **PEC al control:** emergencia observada acumulada hasta la fecha recomendada.
- **Lead time:** días entre la primera alerta y la fecha de control.
- **Ventana 600–800 °Cd:** días calendario disponibles para efectuar el control.
        """
    )
