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
    page_title="Validación — Azul",
    page_icon="📊",
    layout="wide",
)

ARCHIVOS_REQUERIDOS = (
    motor.APP,
    motor.METEO,
    motor.CAMPO,
    *motor.ARCHIVOS_MODELO,
)


def formato_fecha(valor: Any) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    return pd.Timestamp(valor).strftime("%d-%m-%Y")


def formato_numero(
    valor: Any,
    decimales: int = 2,
    sufijo: str = "",
) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    return f"{float(valor):.{decimales}f}{sufijo}"


def huella_archivos(base: Path) -> tuple[tuple[str, int, int], ...]:
    huella: list[tuple[str, int, int]] = []
    for nombre in ARCHIVOS_REQUERIDOS:
        ruta = base / nombre
        if ruta.exists():
            estado = ruta.stat()
            huella.append(
                (nombre, int(estado.st_mtime_ns), int(estado.st_size))
            )
        else:
            huella.append((nombre, -1, -1))
    return tuple(huella)


def ejecutar_sin_interferir_streamlit(
    base: Path,
    campo_acumulado: bool,
) -> dict[str, Any]:
    streamlit_real = sys.modules.get("streamlit")
    try:
        return motor.ejecutar_app(
            base,
            campo_acumulado=campo_acumulado,
        )
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

    globales = ejecutar_sin_interferir_streamlit(
        base,
        campo_acumulado,
    )
    sincronizado = motor.sincronizar_eventos(
        globales["df"],
        globales["df_campo"],
        globales["col_fecha"],
        globales["col_plm2"],
    )
    indicadores = motor.metricas(globales, sincronizado)

    columnas = [
        columna
        for columna in (
            "Fecha",
            "EMERREL",
            "DG",
            "Primer_Pico_Habilitado",
            "EMERREL_ANTES_FILTRO_PRIMER_PICO",
        )
        if columna in globales["df"].columns
    ]
    diario = globales["df"][columnas].copy()
    return indicadores, sincronizado, diario


def grafico_flujos(
    sincronizado: pd.DataFrame,
    indicadores: dict[str, Any],
) -> go.Figure:
    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=sincronizado["Fecha"],
            y=sincronizado["Campo_Relativo"],
            name="Observado",
            marker_color="#60A5FA",
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Observado: %{y:.3f}<extra></extra>"
            ),
        )
    )
    figura.add_trace(
        go.Bar(
            x=sincronizado["Fecha"],
            y=sincronizado["Sim_Relativo"],
            name="Simulado por intervalo",
            marker_color="#166534",
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Simulado: %{y:.3f}<extra></extra>"
            ),
        )
    )

    fecha_obs = indicadores.get("Fecha_primer_pico_observado")
    fecha_sim = indicadores.get("Fecha_primer_pico_simulado")
    for fecha, etiqueta, color in (
        (fecha_obs, "Primer pico observado", "#2563EB"),
        (fecha_sim, "Primer pico diario simulado", "#166534"),
    ):
        if pd.notna(fecha):
            figura.add_vline(
                x=fecha,
                line_color=color,
                line_dash="dot",
                line_width=1.7,
                annotation_text=etiqueta,
                annotation_position="top",
            )

    figura.update_layout(
        title="Flujos relativos por intervalo real de monitoreo",
        barmode="group",
        xaxis_title="Fecha",
        yaxis_title="Flujo relativo",
        height=520,
        hovermode="x unified",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=70, r=30, t=110, b=65),
    )
    figura.update_yaxes(
        gridcolor="rgba(148,163,184,0.25)"
    )
    figura.update_xaxes(showgrid=False)
    return figura


def grafico_acumulado(
    sincronizado: pd.DataFrame,
) -> go.Figure:
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=sincronizado["Fecha"],
            y=sincronizado["Campo_Acumulado"] * 100,
            mode="markers+lines",
            name="Observado",
            marker=dict(size=9, color="#60A5FA"),
            line=dict(color="#60A5FA", width=2.2),
        )
    )
    figura.add_trace(
        go.Scatter(
            x=sincronizado["Fecha"],
            y=sincronizado["Sim_Acumulado"] * 100,
            mode="lines",
            name="Simulado",
            line=dict(color="#166534", width=2.8, dash="dash"),
        )
    )
    figura.update_layout(
        title="Trayectoria acumulada observada y simulada",
        xaxis_title="Fecha",
        yaxis_title="Emergencia acumulada (%)",
        height=500,
        hovermode="x unified",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=70, r=25, t=80, b=65),
    )
    figura.update_yaxes(
        range=[0, 105],
        gridcolor="rgba(148,163,184,0.25)",
    )
    figura.update_xaxes(showgrid=False)
    return figura


def grafico_decision(
    diario: pd.DataFrame,
    indicadores: dict[str, Any],
) -> go.Figure:
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=diario["Fecha"],
            y=diario["EMERREL"],
            mode="lines",
            name="EMERREL diaria",
            line=dict(color="#075FCF", width=2.3),
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "EMERREL: %{y:.3f}<extra></extra>"
            ),
        )
    )

    control = indicadores.get("Fecha_control")
    limite = indicadores.get("Fecha_limite")
    pico_sim = indicadores.get("Fecha_primer_pico_simulado")
    pico_obs = indicadores.get("Fecha_primer_pico_observado")

    if pd.notna(control) and pd.notna(limite):
        figura.add_vrect(
            x0=control,
            x1=limite,
            fillcolor="rgba(34,197,94,0.12)",
            layer="below",
            line_width=0,
            annotation_text="Ventana eficiente",
            annotation_position="top left",
        )

    referencias = (
        (pico_sim, "Pico simulado", "#166534", "dot"),
        (pico_obs, "Pico observado", "#2563EB", "dot"),
        (control, "Control", "#111827", "dash"),
        (limite, "Límite", "#166534", "dash"),
    )
    for fecha, texto, color, estilo in referencias:
        if pd.notna(fecha):
            figura.add_vline(
                x=fecha,
                line_color=color,
                line_dash=estilo,
                line_width=1.6,
                annotation_text=texto,
                annotation_position="top",
            )

    figura.update_layout(
        title="Serie diaria y sincronía del primer pico",
        xaxis_title="Fecha",
        yaxis_title="EMERREL",
        height=540,
        hovermode="x unified",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=70, r=30, t=105, b=65),
    )
    figura.update_yaxes(
        range=[0, 1.05],
        gridcolor="rgba(148,163,184,0.25)",
    )
    figura.update_xaxes(showgrid=False)
    return figura


def crear_excel(
    indicadores: dict[str, Any],
    sincronizado: pd.DataFrame,
    diario: pd.DataFrame,
) -> bytes:
    salida = io.BytesIO()
    limpio = {
        clave: motor.serializar(valor)
        for clave, valor in indicadores.items()
    }
    with pd.ExcelWriter(
        salida,
        engine="xlsxwriter",
        datetime_format="dd-mm-yyyy",
    ) as writer:
        pd.DataFrame([limpio]).to_excel(
            writer,
            sheet_name="Metricas",
            index=False,
        )
        sincronizado.to_excel(
            writer,
            sheet_name="Event_to_Event",
            index=False,
        )
        diario.to_excel(
            writer,
            sheet_name="Serie_Diaria",
            index=False,
        )
    return salida.getvalue()


st.title("📊 Validación automática — Azul")
st.caption(
    "El desfase del primer pico se calcula con la fecha diaria del modelo, "
    "mientras que la frecuencia de picos se evalúa por intervalos Event-to-Event."
)

faltantes = [
    nombre
    for nombre in ARCHIVOS_REQUERIDOS
    if not (BASE / nombre).exists()
]

with st.sidebar:
    st.header("Configuración de validación")
    campo_acumulado = st.checkbox(
        "Los datos de campo son acumulados",
        value=bool(motor.CAMPO_ES_ACUMULADO),
        help=(
            "Activar solo cuando cada conteo sea el total acumulado "
            "desde el inicio."
        ),
    )
    umbral_evento = st.number_input(
        "Umbral de flujo significativo",
        min_value=0.00,
        max_value=1.00,
        value=float(motor.UMBRAL_EVENTO),
        step=0.01,
        format="%.2f",
    )
    prominencia_pico = st.number_input(
        "Prominencia mínima del pico",
        min_value=0.00,
        max_value=1.00,
        value=float(motor.PROMINENCIA_PICO),
        step=0.01,
        format="%.2f",
    )
    recalcular = st.button(
        "🔄 Recalcular métricas",
        type="primary",
        width="stretch",
        disabled=bool(faltantes),
    )
    st.divider()
    st.code(
        f"Modelo: {motor.APP}\n"
        f"Meteorología: {motor.METEO}\n"
        f"Campo: {motor.CAMPO}",
        language=None,
    )

if faltantes:
    st.error(
        "Faltan archivos en el repositorio:\n\n- "
        + "\n- ".join(faltantes)
    )
    st.stop()

if recalcular:
    calcular_resultados.clear()

with st.spinner(
    "Ejecutando PREDWEEM Azul y calculando las métricas..."
):
    try:
        indicadores, sincronizado, diario = calcular_resultados(
            str(BASE),
            campo_acumulado,
            float(umbral_evento),
            float(prominencia_pico),
            huella_archivos(BASE),
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()


delta_diario = indicadores.get("Delta_primer_pico_dias")
delta_intervalo = indicadores.get(
    "Delta_primer_pico_intervalo_dias"
)

if pd.notna(delta_diario) and int(delta_diario) == -9:
    st.success(
        "✅ Sincronía verificada: el primer pico diario simulado "
        "anticipa al observado en 9 días (Δ = −9 días)."
    )
elif pd.notna(delta_diario):
    st.warning(
        "El Δ diario calculado es "
        f"{int(delta_diario):+d} días; para los datos actuales de Azul "
        "se espera −9 días. Revise los datos o parámetros utilizados."
    )

st.subheader("Resumen de desempeño")
resumen = st.columns(6)
resumen[0].metric(
    "F1 de picos",
    formato_numero(indicadores.get("F1_picos"), 2),
)
resumen[1].metric(
    "NSE de flujos",
    formato_numero(indicadores.get("NSE_flujos"), 2),
)
resumen[2].metric(
    "Δ primer pico diario",
    formato_numero(delta_diario, 0, " d"),
    help=(
        "Fecha del primer pico diario validado del modelo menos "
        "fecha del primer pico observado. Negativo = anticipación."
    ),
)
resumen[3].metric(
    "Δ por intervalo",
    formato_numero(delta_intervalo, 0, " d"),
    help=(
        "Resultado de auditoría basado en las fechas finales de los "
        "intervalos Event-to-Event; no sustituye al Δ diario."
    ),
)
resumen[4].metric(
    "PEC al control",
    formato_numero(indicadores.get("PEC_control_pct"), 1, " %"),
)
resumen[5].metric(
    "Ventana 600–800",
    formato_numero(
        indicadores.get("Ventana_600_800_dias"),
        0,
        " d",
    ),
)

picos = st.columns(5)
picos[0].metric(
    "Picos observados",
    formato_numero(indicadores.get("Picos_observados"), 0),
)
picos[1].metric(
    "Picos simulados",
    formato_numero(indicadores.get("Picos_simulados"), 0),
)
picos[2].metric(
    "Picos coincidentes",
    formato_numero(indicadores.get("Hits_picos"), 0),
)
picos[3].metric(
    "Picos omitidos",
    formato_numero(indicadores.get("Omisiones_picos"), 0),
)
picos[4].metric(
    "Falsos picos",
    formato_numero(indicadores.get("Falsos_picos"), 0),
)

st.subheader("Fechas del primer pico y decisión")
fechas = st.columns(5)
fechas[0].metric(
    "Pico diario simulado",
    formato_fecha(indicadores.get("Fecha_primer_pico_simulado")),
)
fechas[1].metric(
    "Pico observado",
    formato_fecha(indicadores.get("Fecha_primer_pico_observado")),
)
fechas[2].metric(
    "Pico simulado por intervalo",
    formato_fecha(
        indicadores.get("Fecha_primer_pico_simulado_intervalo")
    ),
)
fechas[3].metric(
    "Control recomendado",
    formato_fecha(indicadores.get("Fecha_control")),
)
fechas[4].metric(
    "Límite de control",
    formato_fecha(indicadores.get("Fecha_limite")),
)

config_png = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "PREDWEEM_Azul_validacion",
        "width": 1800,
        "height": 1000,
        "scale": 2,
    },
}

tab_flujos, tab_acumulado, tab_decision, tab_descargas = st.tabs(
    [
        "Picos y flujos",
        "Trayectoria acumulada",
        "Decisión de control",
        "Datos y descargas",
    ]
)

with tab_flujos:
    st.plotly_chart(
        grafico_flujos(sincronizado, indicadores),
        width="stretch",
        config=config_png,
    )
    st.dataframe(
        sincronizado,
        width="stretch",
        hide_index=True,
    )

with tab_acumulado:
    st.plotly_chart(
        grafico_acumulado(sincronizado),
        width="stretch",
        config=config_png,
    )

with tab_decision:
    st.plotly_chart(
        grafico_decision(diario, indicadores),
        width="stretch",
        config=config_png,
    )
    st.info(
        "La línea verde marca el primer pico diario simulado; la azul, "
        "el primer pico observado. La franja verde representa la ventana "
        "térmica entre 600 y 800 °Cd."
    )

with tab_descargas:
    limpio = {
        clave: motor.serializar(valor)
        for clave, valor in indicadores.items()
    }
    tabla_metricas = pd.DataFrame([limpio])
    st.dataframe(
        tabla_metricas,
        width="stretch",
        hide_index=True,
    )

    excel = crear_excel(
        indicadores,
        sincronizado,
        diario,
    )
    csv_metricas = tabla_metricas.to_csv(
        index=False
    ).encode("utf-8-sig")
    csv_eventos = sincronizado.to_csv(
        index=False,
        date_format="%Y-%m-%d",
    ).encode("utf-8-sig")
    json_metricas = json.dumps(
        limpio,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "📥 Descargar Excel",
        data=excel,
        file_name="PREDWEEM_validacion_azul.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )
    d2.download_button(
        "📥 Métricas CSV",
        data=csv_metricas,
        file_name="metricas_azul.csv",
        mime="text/csv",
        width="stretch",
    )
    d3.download_button(
        "📥 Métricas JSON",
        data=json_metricas,
        file_name="metricas_azul.json",
        mime="application/json",
        width="stretch",
    )
    st.download_button(
        "📥 Event-to-Event CSV",
        data=csv_eventos,
        file_name="event_to_event_azul.csv",
        mime="text/csv",
        width="stretch",
    )

with st.expander("Definición de los indicadores"):
    st.markdown(
        """
- **Δ primer pico diario:** fecha del primer pico diario validado del modelo menos fecha del primer pico observado. Para Azul, con los datos actuales, debe ser **−9 días**.
- **Δ por intervalo:** diferencia entre las fechas finales de los intervalos que contienen el primer máximo observado y simulado. Se conserva como auditoría y puede ser 0 aunque exista anticipación diaria.
- **F1 de picos:** coincidencia entre máximos locales de los flujos Event-to-Event.
- **NSE de flujos:** ajuste de las magnitudes relativas por intervalo real de monitoreo.
- **PEC al control:** porcentaje de la emergencia observada acumulada hasta la fecha recomendada.
- **Ventana 600–800 °Cd:** días calendario disponibles entre el control recomendado y el límite operativo.
        """
    )
