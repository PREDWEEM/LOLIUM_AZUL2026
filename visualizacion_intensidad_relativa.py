# -*- coding: utf-8 -*-
"""Escala visual relativa 0–100 % para el gráfico principal de PREDWEEM Azul.

La transformación es exclusivamente de presentación. No modifica EMERREL del
motor ni, por lo tanto, Event-to-Event, T50, métricas, umbrales internos o
clasificación. El máximo diario simulado de la campaña se muestra como 100 %.
"""

from __future__ import annotations


def _reemplazar_unico(source: str, old: str, new: str, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != 1:
        raise RuntimeError(
            f"Parche visual Azul no aplicado: '{etiqueta}' aparece {cantidad} veces; "
            "se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def _reemplazar_n(source: str, old: str, new: str, cantidad_esperada: int, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != cantidad_esperada:
        raise RuntimeError(
            f"Parche visual Azul no aplicado: '{etiqueta}' aparece {cantidad} veces; "
            f"se esperaban {cantidad_esperada} coincidencias."
        )
    return source.replace(old, new)


def parchear_visualizacion_intensidad_relativa(source: str) -> str:
    """Reemplaza la escala logarítmica del gráfico principal por 0–100 %."""

    transformacion_old = '''    # Transformación Logarítmica Analítica
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)
    if df_campo is not None:
        df_campo['Campo_Normalizado_LOG'] = np.log10(df_campo['Campo_Normalizado'] + c_log)'''

    transformacion_new = '''    # Escala visual relativa 0–100 %; EMERREL del motor permanece intacto.
    max_emerrel_visual = float(df["EMERREL"].clip(lower=0.0).max())
    if max_emerrel_visual > 0.0:
        df["EMERREL_REL_PCT"] = (
            df["EMERREL"].clip(lower=0.0) / max_emerrel_visual * 100.0
        )
        umbral_er_pct = float(umbral_er) / max_emerrel_visual * 100.0
    else:
        df["EMERREL_REL_PCT"] = 0.0
        umbral_er_pct = 0.0
    if df_campo is not None:
        df_campo["Campo_Normalizado_PCT"] = (
            df_campo["Campo_Normalizado"].clip(lower=0.0) * 100.0
        )'''

    source = _reemplazar_unico(
        source,
        transformacion_old,
        transformacion_new,
        "transformación visual",
    )

    source = _reemplazar_unico(
        source,
        '                    y=df["EMERREL_LOG"],\n                    mode="lines",',
        '                    y=df["EMERREL_REL_PCT"],\n                    customdata=df[["EMERREL"]].to_numpy(),\n                    mode="lines",',
        "serie simulada",
    )
    source = _reemplazar_unico(
        source,
        '                    name="Tasa diaria simulada (log)",',
        '                    name="Emergencia diaria simulada (%)",',
        "nombre serie simulada",
    )
    source = _reemplazar_unico(
        source,
        '                        "Simulado: %{y:.3f}<extra></extra>"',
        '                        "Intensidad relativa: %{y:.1f}%<br>"\n                        "EMERREL: %{customdata[0]:.3f}<extra></extra>"',
        "hover simulado",
    )

    source = _reemplazar_unico(
        source,
        '                        y=df_campo["Campo_Normalizado_LOG"],',
        '                        y=df_campo["Campo_Normalizado_PCT"],',
        "serie campo",
    )
    source = _reemplazar_unico(
        source,
        '                        name="Campo normalizado (log)",',
        '                        name="Campo normalizado (%)",',
        "nombre serie campo",
    )
    source = _reemplazar_unico(
        source,
        '                            "Campo: %{y:.3f}<extra></extra>"',
        '                            "Campo: %{y:.1f}%<extra></extra>"',
        "hover campo",
    )

    source = _reemplazar_n(
        source,
        "umbral_er_log",
        "umbral_er_pct",
        2,
        "umbral gráfico",
    )

    source = _reemplazar_unico(
        source,
        '                        text="Log10(EMERREL + 0,01)",',
        '                        text="Intensidad relativa de emergencia (%)",',
        "título eje Y",
    )
    source = _reemplazar_unico(
        source,
        "                    range=[-2.18, 0.12],",
        "                    range=[0.0, 105.0],",
        "rango eje Y",
    )
    source = _reemplazar_unico(
        source,
        "                    tickvals=[-2.0, -1.5, -1.0, -0.5, 0.0],",
        "                    tickvals=[0, 20, 40, 60, 80, 100],",
        "ticks eje Y",
    )

    return source
