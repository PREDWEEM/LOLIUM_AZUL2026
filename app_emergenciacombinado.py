# -*- coding: utf-8 -*-
"""
PREDWEEM Lolium — Azul.

La página muestra la alerta de grandes flujos de Lolium del paquete ``lolium-comun`` (probabilidad de gran flujo en
los próximos 7 días, qué hacer esta semana, ventana del graminicida de barbecho, residual a la siembra y flujos tardíos)
con el clima diario del repositorio (``meteo_daily.csv``).

El motor PREDWEEM anterior (red neuronal y balance hídrico, ``app_emergenciacombinado_core.py`` y sus modelos) ya no se
ejecuta: su inicio estimado de la emergencia erró 12–14 días en promedio y no mejoraba la alerta. Sus archivos quedan
en el repositorio solo como referencia.
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
from acceso_predweem import mostrar_sesion, requerir_acceso
from lolium_comun import pagina

BASE = Path(__file__).resolve().parent

requerir_acceso("Azul")
pagina.encabezado("🌾 PREDWEEM LOLIUM — AZUL (BA) lat=-36.8700 lon=-59.8900", "PREDWEEM AZUL")
parametros = pagina.barra_lateral(BASE, "Azul")
meteo = pagina.cargar_meteo(BASE, None)

NOTA_AZUL = (
    "**Validación en Azul.** El modelo se probó, sin reajustarlo, con los conteos de Azul 2026 "
    "(11 fechas entre el 01/03 y el 01/09; 10 intervalos de 11 a 29 días; 4 con emergencia ≥ 5 % de la campaña por "
    "semana, todos entre el 01/03 y el 20/05) y el clima de esta app. Emitida al comienzo de cada intervalo, la alerta "
    "ordenó perfectamente los intervalos con y sin gran flujo (AUC 1,00 en los tres niveles de información): los 4 quedaron "
    "en *Alerta* y ninguno de los 6 sin gran flujo. El gemelo digital de Azul no pudo dar porcentajes de emergencia durante "
    "2026, porque su referencia es la propia campaña 2026 y recién está disponible al cerrarla.\n\n"
    "**Límites en Azul.** Una sola campaña y conteos cada 2–4 semanas: con intervalos tan largos, un \"gran flujo\" es en "
    "realidad un período de emergencia sostenida, y el resultado es mucho menos exigente que una validación semanal. El "
    "clima es reanálisis ERA5 en la grilla de Azul, no observaciones de estación. La alerta con conteos está calibrada con "
    "conteos semanales; con recorridas espaciadas conviene usar sobre todo la fecha de primeras plántulas. La curva "
    "acumulada es el calendario de referencia regional (error medio de 6 puntos en Azul 2026); con una segunda campaña de Azul se "
    "podrá usar un calendario propio, como en Tres Arroyos."
)


_resultados_alerta = None
if meteo is not None and not meteo.empty:
    try:
        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        _resultados_alerta = _vis_alerta.mostrar_alerta_lolmu(
            meteo,
            **parametros,
    tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
    curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
    curva_desde_primeras=True,  # con primeras plántulas, la curva del calendario arranca ese día
    localidad="Azul",
    nota_sitio=NOTA_AZUL,
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")

pagina.descarga_excel(_resultados_alerta, "Azul", "descarga_excel_resultados_final_azul")
mostrar_sesion()
