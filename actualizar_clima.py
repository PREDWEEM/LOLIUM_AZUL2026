# -*- coding: utf-8 -*-
"""Actualiza la meteorología diaria de PREDWEEM Azul.

Estrategia de fuentes
---------------------
- ERA5-Land: histórico consolidado hasta seis días antes de la fecha de ejecución.
- ECMWF IFS histórico: cubre los días recientes todavía no disponibles en ERA5-Land.
- ECMWF IFS HRES: pronóstico operativo desde hoy hasta +7 días.

La precipitación faltante nunca se completa con cero ni se arrastra desde el día
anterior. Ante un dato crítico ausente, el script falla y conserva intacto el CSV
existente.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LAT = -36.87
LON = -59.89
TIMEZONE = "America/Argentina/Buenos_Aires"
ARCHIVO_CSV = Path("meteo_daily.csv")
FECHA_INICIO = date(2026, 1, 1)
RETARDO_ERA5_DIAS = 6
REFRESCO_ERA5_DIAS = 7
HORIZONTE_PRONOSTICO_DIAS = 8  # hoy + 7 días
TIMEOUT_SEGUNDOS = 60

VARIABLES_DIARIAS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
]
COLUMNAS_SALIDA = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
    "FUENTE",
    "TIPO",
    "FECHA_EMISION",
]


def _ahora_local() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def _fecha_iso(valor: date) -> str:
    return valor.isoformat()


def _consultar_api(
    url: str,
    params: dict,
    *,
    fuente: str,
    tipo: str | Callable[[date], str],
    fecha_emision: str,
) -> pd.DataFrame:
    """Consulta Open-Meteo y devuelve un bloque diario validado."""
    respuesta = requests.get(url, params=params, timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()
    payload = respuesta.json()

    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo devolvió un error: {payload.get('reason', payload)}")
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
        raise RuntimeError("Variables faltantes en la respuesta: " + ", ".join(sorted(faltantes)))

    longitudes = {clave: len(daily[clave]) for clave in requeridas}
    if len(set(longitudes.values())) != 1:
        raise RuntimeError(f"Longitudes inconsistentes en la respuesta diaria: {longitudes}")

    df = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(daily["time"], errors="coerce"),
            "TMAX": pd.to_numeric(daily["temperature_2m_max"], errors="coerce"),
            "TMIN": pd.to_numeric(daily["temperature_2m_min"], errors="coerce"),
            "Prec": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
        }
    )

    # Solo se admite interpolar un único hueco interior de temperatura.
    # La precipitación jamás se completa, ni con cero ni por arrastre.
    for columna in ("TMAX", "TMIN"):
        df[columna] = df[columna].interpolate(limit=1, limit_area="inside")

    df["Fecha"] = df["Fecha"].dt.normalize()
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates("Fecha", keep="last")

    nulos = df[["TMAX", "TMIN", "Prec"]].isna().any(axis=1)
    if nulos.any():
        fechas = ", ".join(df.loc[nulos, "Fecha"].dt.strftime("%Y-%m-%d").tolist())
        raise RuntimeError(
            "Datos meteorológicos críticos ausentes en: "
            f"{fechas}. No se modificó {ARCHIVO_CSV}."
        )

    if (df["TMAX"] < df["TMIN"]).any():
        fechas = ", ".join(
            df.loc[df["TMAX"] < df["TMIN"], "Fecha"].dt.strftime("%Y-%m-%d").tolist()
        )
        raise RuntimeError(f"TMAX menor que TMIN en: {fechas}")
    if (df["Prec"] < 0).any():
        fechas = ", ".join(
            df.loc[df["Prec"] < 0, "Fecha"].dt.strftime("%Y-%m-%d").tolist()
        )
        raise RuntimeError(f"Precipitación negativa en: {fechas}")

    df["FUENTE"] = fuente
    if callable(tipo):
        df["TIPO"] = [tipo(ts.date()) for ts in df["Fecha"]]
    else:
        df["TIPO"] = tipo
    df["FECHA_EMISION"] = fecha_emision
    return df[COLUMNAS_SALIDA]


def _descargar_historico(
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

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": _fecha_iso(fecha_desde),
        "end_date": _fecha_iso(fecha_hasta),
        "daily": VARIABLES_DIARIAS,
        "models": modelo,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    print(f"Descargando {fuente}: {fecha_desde} a {fecha_hasta}...")
    return _consultar_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params,
        fuente=fuente,
        tipo=tipo,
        fecha_emision=fecha_emision,
    )


def _descargar_pronostico(hoy: date, fecha_emision: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": VARIABLES_DIARIAS,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "forecast_days": HORIZONTE_PRONOSTICO_DIAS,
        "cell_selection": "land",
    }

    print(f"Descargando ECMWF IFS HRES: {hoy} a hoy +7 días...")
    return _consultar_api(
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


def _bloque_era5_congelado(df_existente: pd.DataFrame, fecha_refresco: date) -> pd.DataFrame:
    """Conserva ERA5-Land antiguo y evita reescribir todo el historial en cada ejecución."""
    if df_existente.empty:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    mascara = (
        (df_existente["FUENTE"] == "ERA5_LAND")
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
        muestra = ", ".join(ts.strftime("%Y-%m-%d") for ts in faltantes[:15])
        extra = "..." if len(faltantes) > 15 else ""
        raise RuntimeError(f"La serie final tiene {len(faltantes)} fechas faltantes: {muestra}{extra}")

    if df["Fecha"].duplicated().any():
        raise RuntimeError("La serie final contiene fechas duplicadas.")
    if df[["TMAX", "TMIN", "Prec"]].isna().any().any():
        raise RuntimeError("La serie final contiene valores meteorológicos nulos.")


def actualizar_meteorologia() -> pd.DataFrame:
    ahora = _ahora_local()
    hoy = ahora.date()
    fecha_emision = ahora.isoformat(timespec="seconds")

    era5_hasta = hoy - timedelta(days=RETARDO_ERA5_DIAS)
    ifs_desde = max(FECHA_INICIO, era5_hasta + timedelta(days=1))
    ifs_hasta = hoy - timedelta(days=1)
    fecha_refresco_era5 = max(
        FECHA_INICIO,
        era5_hasta - timedelta(days=REFRESCO_ERA5_DIAS - 1),
    )

    existente = _leer_existente()
    congelado = _bloque_era5_congelado(existente, fecha_refresco_era5)

    # Primera migración: reconstruye todo el histórico consolidado con ERA5-Land.
    # Ejecuciones posteriores: solo refresca la cola reciente de ERA5-Land.
    era5_desde = FECHA_INICIO if congelado.empty else fecha_refresco_era5
    era5 = _descargar_historico(
        era5_desde,
        era5_hasta,
        modelo="era5_land",
        fuente="ERA5_LAND",
        tipo="REANALISIS",
        fecha_emision=fecha_emision,
    )

    ifs_reciente = _descargar_historico(
        ifs_desde,
        ifs_hasta,
        modelo="ecmwf_ifs",
        fuente="ECMWF_IFS",
        tipo="HISTORICO_MODELO",
        fecha_emision=fecha_emision,
    )

    pronostico = _descargar_pronostico(hoy, fecha_emision)

    bloques = [bloque for bloque in (congelado, era5, ifs_reciente, pronostico) if not bloque.empty]
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

    # Escritura atómica: el archivo operativo solo se reemplaza después de validar todo.
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
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
