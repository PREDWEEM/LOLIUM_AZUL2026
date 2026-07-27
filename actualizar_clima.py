# -*- coding: utf-8 -*-
"""Actualiza la meteorología diaria de PREDWEEM Azul.

Estrategia de fuentes
---------------------
- ERA5-Land: histórico consolidado hasta su último día realmente completo.
- ERA5: respaldo explícito si ERA5-Land no entrega una ventana reciente válida.
- ECMWF IFS histórico: cubre los días posteriores al último reanálisis completo.
- ECMWF IFS HRES: pronóstico operativo desde hoy hasta +7 días.

La precipitación faltante nunca se completa con cero ni se arrastra desde el día
anterior. Antes de descargar el bloque histórico completo, el script sondea una
ventana reciente para determinar el último día con Tmax, Tmin y precipitación
válidas. Así evita pedir como reanálisis un día que todavía está incompleto y
transfiere automáticamente ese tramo reciente a ECMWF IFS.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LAT = -36.87
LON = -59.89
TIMEZONE = "America/Argentina/Buenos_Aires"
ARCHIVO_CSV = Path("meteo_daily.csv")
FECHA_INICIO = date(2026, 1, 1)
RETARDO_ERA5_OBJETIVO_DIAS = 5
VENTANA_SONDEO_REANALISIS_DIAS = 21
REFRESCO_ERA5_DIAS = 7
HORIZONTE_PRONOSTICO_DIAS = 8  # hoy + 7 días
TIMEOUT_SEGUNDOS = 90

VARIABLES_DIARIAS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
)
VARIABLES_HORARIAS = (
    "temperature_2m",
    "precipitation",
)
COLUMNAS_SALIDA = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
    "FUENTE",
    "TIPO",
    "FECHA_EMISION",
]


class DatosMeteorologicosAusentes(RuntimeError):
    """La API respondió, pero uno o más valores meteorológicos son nulos."""


def _ahora_local() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def _fecha_iso(valor: date) -> str:
    return valor.isoformat()


def _lista_api(valores: tuple[str, ...]) -> str:
    """Serializa arrays según OpenAPI (explode=false): valor1,valor2,..."""
    return ",".join(valores)


def _resumen_fechas(fechas: pd.Series | pd.DatetimeIndex, limite: int = 15) -> str:
    serie = pd.DatetimeIndex(pd.to_datetime(fechas, errors="coerce")).dropna()
    muestra = ", ".join(ts.strftime("%Y-%m-%d") for ts in serie[:limite])
    if len(serie) > limite:
        muestra += f", ... ({len(serie)} fechas en total)"
    return muestra


def _get_json(url: str, params: dict) -> dict:
    respuesta = requests.get(url, params=params, timeout=TIMEOUT_SEGUNDOS)
    print(f"URL consultada: {respuesta.url}")
    respuesta.raise_for_status()
    payload = respuesta.json()
    if payload.get("error"):
        raise RuntimeError(
            f"Open-Meteo devolvió un error: {payload.get('reason', payload)}"
        )
    return payload


def _agregar_metadatos(
    df: pd.DataFrame,
    *,
    fuente: str,
    tipo: str,
    fecha_emision: str,
) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
    for columna in ("TMAX", "TMIN", "Prec"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = (
        df.dropna(subset=["Fecha"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )

    # Solo se admite interpolar un único hueco interior de temperatura.
    # La precipitación jamás se completa, ni con cero ni por arrastre.
    for columna in ("TMAX", "TMIN"):
        df[columna] = df[columna].interpolate(limit=1, limit_area="inside")

    nulos = df[["TMAX", "TMIN", "Prec"]].isna().any(axis=1)
    if nulos.any():
        raise DatosMeteorologicosAusentes(
            "Datos meteorológicos críticos ausentes en: "
            + _resumen_fechas(df.loc[nulos, "Fecha"])
        )

    if (df["TMAX"] < df["TMIN"]).any():
        fechas = df.loc[df["TMAX"] < df["TMIN"], "Fecha"]
        raise RuntimeError("TMAX menor que TMIN en: " + _resumen_fechas(fechas))

    if (df["Prec"] < 0).any():
        fechas = df.loc[df["Prec"] < 0, "Fecha"]
        raise RuntimeError("Precipitación negativa en: " + _resumen_fechas(fechas))

    df["FUENTE"] = fuente
    df["TIPO"] = tipo
    df["FECHA_EMISION"] = fecha_emision
    return df[COLUMNAS_SALIDA]


def _extraer_diario_sin_validar(payload: dict) -> pd.DataFrame:
    """Convierte el bloque diario en tabla sin rechazar todavía los nulos."""
    if "daily" not in payload:
        raise RuntimeError(f"Respuesta sin bloque 'daily': {payload}")

    daily = payload["daily"]
    requeridas = {
        "time",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    }
    faltantes = requeridas.difference(daily)
    if faltantes:
        raise RuntimeError(
            "Variables faltantes en la respuesta diaria: "
            + ", ".join(sorted(faltantes))
        )

    longitudes = {clave: len(daily[clave]) for clave in requeridas}
    if len(set(longitudes.values())) != 1:
        raise RuntimeError(
            f"Longitudes inconsistentes en la respuesta diaria: {longitudes}"
        )

    return pd.DataFrame(
        {
            "Fecha": pd.to_datetime(daily["time"], errors="coerce"),
            "TMAX": pd.to_numeric(daily["temperature_2m_max"], errors="coerce"),
            "TMIN": pd.to_numeric(daily["temperature_2m_min"], errors="coerce"),
            "Prec": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
        }
    ).dropna(subset=["Fecha"])


def _consultar_diario(
    url: str,
    params: dict,
    *,
    fuente: str,
    tipo: str,
    fecha_emision: str,
) -> pd.DataFrame:
    payload = _get_json(url, params)
    df = _extraer_diario_sin_validar(payload)
    return _agregar_metadatos(
        df,
        fuente=fuente,
        tipo=tipo,
        fecha_emision=fecha_emision,
    )


def _consultar_horario_agregado(
    url: str,
    params: dict,
    *,
    fecha_desde: date,
    fecha_hasta: date,
    fuente: str,
    tipo: str,
    fecha_emision: str,
) -> pd.DataFrame:
    """Agrega ERA5/IFS horario a Tmax, Tmin y precipitación diaria local."""
    payload = _get_json(url, params)
    if "hourly" not in payload:
        raise RuntimeError(f"Respuesta sin bloque 'hourly': {payload}")

    hourly = payload["hourly"]
    requeridas = {"time", "temperature_2m", "precipitation"}
    faltantes = requeridas.difference(hourly)
    if faltantes:
        raise RuntimeError(
            "Variables faltantes en la respuesta horaria: "
            + ", ".join(sorted(faltantes))
        )

    bruto = pd.DataFrame(
        {
            "Hora": pd.to_datetime(hourly["time"], errors="coerce"),
            "Temperatura": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "Precipitacion": pd.to_numeric(hourly["precipitation"], errors="coerce"),
        }
    ).dropna(subset=["Hora"])

    nulos = bruto[["Temperatura", "Precipitacion"]].isna().any(axis=1)
    if nulos.any():
        raise DatosMeteorologicosAusentes(
            "La respuesta horaria también contiene valores nulos en: "
            + _resumen_fechas(bruto.loc[nulos, "Hora"])
        )

    bruto["Fecha"] = bruto["Hora"].dt.normalize()
    agregado = (
        bruto.groupby("Fecha", as_index=False)
        .agg(
            TMAX=("Temperatura", "max"),
            TMIN=("Temperatura", "min"),
            Prec=("Precipitacion", "sum"),
            Horas_T=("Temperatura", "count"),
            Horas_P=("Precipitacion", "count"),
        )
    )

    esperadas = pd.date_range(fecha_desde, fecha_hasta, freq="D")
    faltan_dias = esperadas.difference(pd.DatetimeIndex(agregado["Fecha"]))
    if len(faltan_dias):
        raise DatosMeteorologicosAusentes(
            "Faltan días en el respaldo horario: " + _resumen_fechas(faltan_dias)
        )

    # Argentina no utiliza horario de verano: cada día completo debe tener 24 horas.
    incompletos = agregado[(agregado["Horas_T"] < 24) | (agregado["Horas_P"] < 24)]
    if not incompletos.empty:
        raise DatosMeteorologicosAusentes(
            "Días horarios incompletos: " + _resumen_fechas(incompletos["Fecha"])
        )

    return _agregar_metadatos(
        agregado[["Fecha", "TMAX", "TMIN", "Prec"]],
        fuente=fuente,
        tipo=tipo,
        fecha_emision=fecha_emision,
    )


def _parametros_historicos(
    fecha_desde: date,
    fecha_hasta: date,
    modelo: str,
) -> dict:
    return {
        "latitude": LAT,
        "longitude": LON,
        "start_date": _fecha_iso(fecha_desde),
        "end_date": _fecha_iso(fecha_hasta),
        "models": modelo,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }


def _ultimo_dia_completo_modelo(
    modelo: str,
    fecha_objetivo: date,
) -> date | None:
    """Detecta el último día diario completo dentro de una ventana reciente."""
    fecha_desde = max(
        FECHA_INICIO,
        fecha_objetivo - timedelta(days=VENTANA_SONDEO_REANALISIS_DIAS - 1),
    )
    params = {
        **_parametros_historicos(fecha_desde, fecha_objetivo, modelo),
        "daily": _lista_api(VARIABLES_DIARIAS),
    }
    print(
        f"Sondeando disponibilidad de {modelo}: "
        f"{fecha_desde} a {fecha_objetivo}..."
    )
    payload = _get_json("https://archive-api.open-meteo.com/v1/archive", params)
    df = _extraer_diario_sin_validar(payload)
    completos = df[
        df[["TMAX", "TMIN", "Prec"]].notna().all(axis=1)
        & (df["TMAX"] >= df["TMIN"])
        & (df["Prec"] >= 0)
    ]
    if completos.empty:
        return None
    return pd.Timestamp(completos["Fecha"].max()).date()


def _resolver_corte_reanalisis(
    fecha_objetivo: date,
) -> tuple[date, str, str, str]:
    """Elige fuente y último día completo de reanálisis, sin inventar valores."""
    candidatos = (
        ("era5_land", "ERA5_LAND", "REANALISIS"),
        ("era5", "ERA5", "REANALISIS_FALLBACK"),
    )
    errores: list[str] = []

    for modelo, fuente, tipo in candidatos:
        try:
            ultimo = _ultimo_dia_completo_modelo(modelo, fecha_objetivo)
        except Exception as exc:
            errores.append(f"{modelo}: {exc}")
            print(f"ADVERTENCIA: no se pudo sondear {modelo}: {exc}")
            continue

        if ultimo is None:
            errores.append(f"{modelo}: sin días completos en la ventana")
            print(f"ADVERTENCIA: {modelo} no presentó días completos recientes.")
            continue

        retraso_real = (fecha_objetivo - ultimo).days
        if retraso_real > 0:
            print(
                f"ADVERTENCIA: {fuente} está incompleto hasta {fecha_objetivo}; "
                f"se usará hasta {ultimo} y ECMWF IFS cubrirá los "
                f"{retraso_real} día(s) posterior(es)."
            )
        else:
            print(f"{fuente} disponible y completo hasta {ultimo}.")
        return ultimo, modelo, fuente, tipo

    raise DatosMeteorologicosAusentes(
        "No se encontró ningún día completo reciente de ERA5-Land ni ERA5. "
        + " | ".join(errores)
    )


def _descargar_historico_modelo(
    fecha_desde: date,
    fecha_hasta: date,
    *,
    modelo: str,
    fuente: str,
    tipo: str,
    fecha_emision: str,
) -> pd.DataFrame:
    if fecha_desde > fecha_hasta:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    base_params = _parametros_historicos(fecha_desde, fecha_hasta, modelo)
    url = "https://archive-api.open-meteo.com/v1/archive"

    print(f"Descargando {fuente} diario: {fecha_desde} a {fecha_hasta}...")
    params_diarios = {**base_params, "daily": _lista_api(VARIABLES_DIARIAS)}
    try:
        return _consultar_diario(
            url,
            params_diarios,
            fuente=fuente,
            tipo=tipo,
            fecha_emision=fecha_emision,
        )
    except DatosMeteorologicosAusentes as exc:
        print(f"ADVERTENCIA: {exc}")
        print(f"Reintentando {fuente} con datos horarios y agregación diaria local...")

    params_horarios = {**base_params, "hourly": _lista_api(VARIABLES_HORARIAS)}
    return _consultar_horario_agregado(
        url,
        params_horarios,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        fuente=fuente,
        tipo=tipo,
        fecha_emision=fecha_emision,
    )


def _descargar_pronostico(hoy: date, fecha_emision: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": _lista_api(VARIABLES_DIARIAS),
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "forecast_days": HORIZONTE_PRONOSTICO_DIAS,
        "cell_selection": "land",
    }

    print(f"Descargando ECMWF IFS HRES: {hoy} a hoy +7 días...")
    return _consultar_diario(
        "https://api.open-meteo.com/v1/ecmwf",
        params,
        fuente="ECMWF_IFS_HRES",
        tipo="PRONOSTICO",
        fecha_emision=fecha_emision,
    )


def _leer_existente() -> pd.DataFrame:
    if not ARCHIVO_CSV.exists():
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    try:
        df = pd.read_csv(ARCHIVO_CSV)
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer {ARCHIVO_CSV}: {exc}") from exc

    if "Fecha" not in df.columns:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
    for columna in ("TMAX", "TMIN", "Prec"):
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
        else:
            df[columna] = pd.NA

    for columna in ("FUENTE", "TIPO", "FECHA_EMISION"):
        if columna not in df.columns:
            df[columna] = pd.NA

    return (
        df[COLUMNAS_SALIDA]
        .dropna(subset=["Fecha"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )


def _bloque_reanalisis_congelado(
    df_existente: pd.DataFrame,
    fecha_refresco: date,
) -> pd.DataFrame:
    """Conserva el reanálisis antiguo y evita reescribir todo el historial."""
    if df_existente.empty:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    tipos_reanalisis = {"REANALISIS", "REANALISIS_FALLBACK"}
    mascara = (
        df_existente["TIPO"].isin(tipos_reanalisis)
        & (df_existente["Fecha"].dt.date < fecha_refresco)
        & (df_existente["Fecha"].dt.date >= FECHA_INICIO)
    )
    return df_existente.loc[mascara, COLUMNAS_SALIDA].copy()


def _validar_continuidad(df: pd.DataFrame, fecha_final: date) -> None:
    if df.empty:
        raise RuntimeError("La actualización produjo una tabla meteorológica vacía.")

    esperadas = pd.date_range(FECHA_INICIO, fecha_final, freq="D")
    disponibles = pd.DatetimeIndex(df["Fecha"].dropna().unique()).normalize()
    faltantes = esperadas.difference(disponibles)
    if len(faltantes):
        raise RuntimeError(
            f"La serie final tiene {len(faltantes)} fechas faltantes: "
            + _resumen_fechas(faltantes)
        )

    if df["Fecha"].duplicated().any():
        raise RuntimeError("La serie final contiene fechas duplicadas.")
    if df[["TMAX", "TMIN", "Prec"]].isna().any().any():
        raise RuntimeError("La serie final contiene valores meteorológicos nulos.")


def actualizar_meteorologia() -> pd.DataFrame:
    ahora = _ahora_local()
    hoy = ahora.date()
    fecha_emision = ahora.isoformat(timespec="seconds")

    fecha_objetivo_reanalisis = hoy - timedelta(days=RETARDO_ERA5_OBJETIVO_DIAS)
    (
        reanalisis_hasta,
        modelo_reanalisis,
        fuente_reanalisis,
        tipo_reanalisis,
    ) = _resolver_corte_reanalisis(fecha_objetivo_reanalisis)

    ifs_desde = max(FECHA_INICIO, reanalisis_hasta + timedelta(days=1))
    ifs_hasta = hoy - timedelta(days=1)
    fecha_refresco_reanalisis = max(
        FECHA_INICIO,
        reanalisis_hasta - timedelta(days=REFRESCO_ERA5_DIAS - 1),
    )

    existente = _leer_existente()
    congelado = _bloque_reanalisis_congelado(
        existente,
        fecha_refresco_reanalisis,
    )

    # Primera migración: reconstruye todo el histórico consolidado.
    # Ejecuciones posteriores: solo refresca la cola reciente del reanálisis.
    reanalisis_desde = (
        FECHA_INICIO if congelado.empty else fecha_refresco_reanalisis
    )
    reanalisis = _descargar_historico_modelo(
        reanalisis_desde,
        reanalisis_hasta,
        modelo=modelo_reanalisis,
        fuente=fuente_reanalisis,
        tipo=tipo_reanalisis,
        fecha_emision=fecha_emision,
    )

    ifs_reciente = _descargar_historico_modelo(
        ifs_desde,
        ifs_hasta,
        modelo="ecmwf_ifs",
        fuente="ECMWF_IFS",
        tipo="HISTORICO_MODELO",
        fecha_emision=fecha_emision,
    )

    pronostico = _descargar_pronostico(hoy, fecha_emision)

    bloques = [
        bloque
        for bloque in (congelado, reanalisis, ifs_reciente, pronostico)
        if not bloque.empty
    ]
    df_final = pd.concat(bloques, ignore_index=True)
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"], errors="coerce").dt.normalize()
    df_final = (
        df_final.sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )

    fecha_final = hoy + timedelta(days=HORIZONTE_PRONOSTICO_DIAS - 1)
    df_final = df_final[
        (df_final["Fecha"].dt.date >= FECHA_INICIO)
        & (df_final["Fecha"].dt.date <= fecha_final)
    ].copy()

    _validar_continuidad(df_final, fecha_final)

    # Escritura atómica: solo reemplaza el archivo después de validar todo.
    temporal = ARCHIVO_CSV.with_suffix(".csv.tmp")
    salida = df_final.copy()
    salida["Fecha"] = salida["Fecha"].dt.strftime("%Y-%m-%d")
    salida.to_csv(temporal, index=False, float_format="%.1f")
    os.replace(temporal, ARCHIVO_CSV)

    print("Actualización meteorológica completada.")
    print(salida.groupby(["FUENTE", "TIPO"], dropna=False).size().to_string())
    print("Últimos registros:")
    print(salida.tail(10).to_string(index=False))
    return salida


def main() -> None:
    try:
        actualizar_meteorologia()
    except Exception as exc:
        print(f"ERROR: {exc}. No se modificó {ARCHIVO_CSV}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
