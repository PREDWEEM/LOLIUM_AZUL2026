# -*- coding: utf-8 -*-
"""Calcula automáticamente las métricas de validación de PREDWEEM Azul.

El motor no se duplica: se ejecuta ``app_emergenciacombinado.py`` en modo
silencioso con los mismos archivos y parámetros predeterminados de Streamlit.
Luego se recalcula una integración Event-to-Event común —incluido el primer
muestreo— y se escriben resultados estandarizados para la matriz multisitio.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SITIO = "Azul"
ID_SITIO = "azul"
APP = "app_emergenciacombinado.py"
METEO = "meteo_daily.csv"
CAMPO = "VALIDA (1).xlsx"
# Nombre que la app busca cuando no se carga un archivo manual.
CAMPO_ALIAS_APP = "azul_campo.xlsx"
SALIDA = "resultados_validacion"
UMBRAL_EVENTO = 0.05
PROMINENCIA_PICO = 0.02
CAMPO_ES_ACUMULADO = False

ARCHIVOS_MODELO = (
    "IW.npy",
    "bias_IW.npy",
    "LW.npy",
    "bias_out.npy",
    "modelo_clusters_k3.pkl",
)


class SessionState(dict):
    def __getattr__(self, nombre: str) -> Any:
        try:
            return self[nombre]
        except KeyError as exc:
            raise AttributeError(nombre) from exc

    def __setattr__(self, nombre: str, valor: Any) -> None:
        self[nombre] = valor


class ContextoNulo:
    def __enter__(self) -> "ContextoNulo":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def __getattr__(self, _: str):
        return lambda *args, **kwargs: None


class StreamlitSilencioso(ContextoNulo):
    def __init__(self) -> None:
        self.session_state = SessionState()
        self.sidebar = self

    @staticmethod
    def _valor(args: tuple[Any, ...], kwargs: dict[str, Any], defecto: Any = 0) -> Any:
        if "value" in kwargs:
            return kwargs["value"]
        # Después de label: min_value, max_value, value, step.
        if len(args) >= 3:
            return args[2]
        return defecto

    def slider(self, label: str, *args: Any, **kwargs: Any) -> Any:
        return self._valor(args, kwargs)

    def number_input(self, label: str, *args: Any, **kwargs: Any) -> Any:
        return self._valor(args, kwargs)

    def checkbox(self, label: str, *args: Any, **kwargs: Any) -> bool:
        if "value" in kwargs:
            return bool(kwargs["value"])
        return bool(args[0]) if args else False

    def file_uploader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def columns(self, especificacion: Any, *args: Any, **kwargs: Any) -> list[ContextoNulo]:
        cantidad = especificacion if isinstance(especificacion, int) else len(especificacion)
        return [ContextoNulo() for _ in range(cantidad)]

    def tabs(self, nombres: Iterable[str]) -> list[ContextoNulo]:
        return [ContextoNulo() for _ in nombres]

    def container(self, *args: Any, **kwargs: Any) -> ContextoNulo:
        return ContextoNulo()

    def expander(self, *args: Any, **kwargs: Any) -> ContextoNulo:
        return ContextoNulo()

    def progress(self, *args: Any, **kwargs: Any) -> ContextoNulo:
        return ContextoNulo()

    def spinner(self, *args: Any, **kwargs: Any) -> ContextoNulo:
        return ContextoNulo()

    def cache_resource(self, funcion=None, **kwargs: Any):
        if funcion is None:
            return lambda f: f
        return funcion

    def rerun(self) -> None:
        return None

    def stop(self) -> None:
        raise RuntimeError("La aplicación invocó st.stop().")

    def __getattr__(self, _: str):
        return lambda *args, **kwargs: None


def instalar_streamlit_silencioso() -> None:
    stub = StreamlitSilencioso()
    modulo = types.ModuleType("streamlit")
    for nombre in dir(stub):
        if not nombre.startswith("_"):
            setattr(modulo, nombre, getattr(stub, nombre))
    modulo.sidebar = stub
    modulo.session_state = stub.session_state
    modulo.__getattr__ = stub.__getattr__
    sys.modules["streamlit"] = modulo


def ejecutar_app(base: Path, campo_acumulado: bool) -> dict[str, Any]:
    requeridos = [APP, METEO, CAMPO, *ARCHIVOS_MODELO]
    faltantes = [nombre for nombre in requeridos if not (base / nombre).exists()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos: " + ", ".join(faltantes))

    with tempfile.TemporaryDirectory(prefix="predweem_validacion_") as temporal:
        destino = Path(temporal)
        for nombre in [APP, METEO, *ARCHIVOS_MODELO]:
            shutil.copy2(base / nombre, destino / nombre)
        shutil.copy2(base / CAMPO, destino / CAMPO_ALIAS_APP)

        instalar_streamlit_silencioso()
        globales = runpy.run_path(str(destino / APP), run_name="__predweem_validacion__")

    claves = ("df", "df_campo", "col_fecha", "col_plm2")
    faltantes_globales = [clave for clave in claves if clave not in globales]
    if faltantes_globales:
        raise RuntimeError(
            "La app no produjo las variables necesarias: " + ", ".join(faltantes_globales)
        )

    campo = globales["df_campo"].copy()
    flujo = globales["col_plm2"]
    campo[flujo] = pd.to_numeric(campo[flujo], errors="coerce").fillna(0).clip(lower=0)
    if campo_acumulado:
        acumulado = campo[flujo].copy()
        campo[flujo] = acumulado.diff().fillna(acumulado).clip(lower=0)
    globales["df_campo"] = campo
    return globales


def sincronizar_eventos(
    diario: pd.DataFrame,
    campo: pd.DataFrame,
    fecha_campo: str,
    flujo_campo: str,
) -> pd.DataFrame:
    sim = diario[["Fecha", "EMERREL"]].copy()
    sim["Fecha"] = pd.to_datetime(sim["Fecha"], errors="coerce")
    sim["EMERREL"] = pd.to_numeric(sim["EMERREL"], errors="coerce").fillna(0).clip(lower=0)
    sim = sim.dropna(subset=["Fecha"]).sort_values("Fecha")

    obs = campo[[fecha_campo, flujo_campo]].copy()
    obs[fecha_campo] = pd.to_datetime(obs[fecha_campo], errors="coerce")
    obs[flujo_campo] = pd.to_numeric(obs[flujo_campo], errors="coerce").fillna(0).clip(lower=0)
    obs = obs.dropna(subset=[fecha_campo]).sort_values(fecha_campo).reset_index(drop=True)
    if sim.empty or obs.empty:
        raise ValueError("No hay datos suficientes para sincronizar.")

    sim = sim[sim["Fecha"] <= obs[fecha_campo].max()]
    inicio_serie = sim["Fecha"].min() - pd.Timedelta(days=1)
    filas: list[dict[str, Any]] = []
    for i, fila in obs.iterrows():
        fin = pd.Timestamp(fila[fecha_campo])
        inicio = inicio_serie if i == 0 else pd.Timestamp(obs.loc[i - 1, fecha_campo])
        mascara = (sim["Fecha"] > inicio) & (sim["Fecha"] <= fin)
        filas.append(
            {
                "Fecha": fin,
                "Inicio_Intervalo": inicio,
                "Dias_Intervalo": max(1, int((fin - inicio).days)),
                "Flujo_Obs_Abs": float(fila[flujo_campo]),
                "Flujo_Sim_Abs": float(sim.loc[mascara, "EMERREL"].sum()),
            }
        )

    salida = pd.DataFrame(filas)
    total_obs = salida["Flujo_Obs_Abs"].sum()
    total_sim = salida["Flujo_Sim_Abs"].sum()
    salida["Campo_Relativo"] = salida["Flujo_Obs_Abs"] / total_obs if total_obs > 0 else 0.0
    salida["Sim_Relativo"] = salida["Flujo_Sim_Abs"] / total_sim if total_sim > 0 else 0.0
    salida["Campo_Acumulado"] = salida["Campo_Relativo"].cumsum()
    salida["Sim_Acumulado"] = salida["Sim_Relativo"].cumsum()
    return salida


def detectar_picos(valores: Iterable[float]) -> list[int]:
    x = np.nan_to_num(np.asarray(list(valores), dtype=float), nan=0.0)
    picos: list[int] = []
    for i, valor in enumerate(x):
        izquierda = x[i - 1] if i > 0 else 0.0
        derecha = x[i + 1] if i + 1 < len(x) else 0.0
        maximo = valor >= izquierda and valor >= derecha and (valor > izquierda or valor > derecha)
        if maximo and valor >= UMBRAL_EVENTO and valor - max(izquierda, derecha) >= PROMINENCIA_PICO:
            picos.append(i)
    return picos


def aparear_picos(obs: list[pd.Timestamp], sim: list[pd.Timestamp], tolerancia: int) -> dict[str, Any]:
    disponibles = set(range(len(sim)))
    hits = 0
    for fecha_obs in obs:
        candidatos = [i for i in disponibles if abs((sim[i] - fecha_obs).days) <= tolerancia]
        if candidatos:
            mejor = min(candidatos, key=lambda i: abs((sim[i] - fecha_obs).days))
            disponibles.remove(mejor)
            hits += 1
    omisiones = len(obs) - hits
    falsos = len(sim) - hits
    precision = hits / (hits + falsos) if hits + falsos else 0.0
    sensibilidad = hits / (hits + omisiones) if hits + omisiones else 0.0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if precision + sensibilidad else 0.0
    return {
        "Hits_picos": hits,
        "Omisiones_picos": omisiones,
        "Falsos_picos": falsos,
        "F1_picos": f1,
    }


def primera_fecha(fechas: pd.Series, valores: pd.Series, umbral: float) -> pd.Timestamp:
    fechas = pd.to_datetime(fechas, errors="coerce")
    valores = pd.to_numeric(valores, errors="coerce").fillna(0)
    candidatas = fechas[valores >= umbral].dropna()
    return pd.Timestamp(candidatas.iloc[0]) if not candidatas.empty else pd.NaT


def metricas(globales: dict[str, Any], sync: pd.DataFrame) -> dict[str, Any]:
    diario = globales["df"]
    campo = globales["df_campo"]
    fecha_campo = globales["col_fecha"]
    flujo_campo = globales["col_plm2"]

    obs = sync["Campo_Relativo"].to_numpy(float)
    sim = sync["Sim_Relativo"].to_numpy(float)
    den = np.sum((obs - obs.mean()) ** 2)
    nse = float(1 - np.sum((sim - obs) ** 2) / den) if den > 0 else np.nan
    pearson = float(np.corrcoef(obs, sim)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(sim) > 0 else np.nan

    eventos_obs = obs >= UMBRAL_EVENTO
    eventos_sim = sim >= UMBRAL_EVENTO
    hits_i = int(np.sum(eventos_obs & eventos_sim))
    omisiones_i = int(np.sum(eventos_obs & ~eventos_sim))
    falsos_i = int(np.sum(~eventos_obs & eventos_sim))
    precision_i = hits_i / (hits_i + falsos_i) if hits_i + falsos_i else 0.0
    sensibilidad_i = hits_i / (hits_i + omisiones_i) if hits_i + omisiones_i else 0.0
    f1_i = 2 * precision_i * sensibilidad_i / (precision_i + sensibilidad_i) if precision_i + sensibilidad_i else 0.0

    indices_obs = detectar_picos(sync["Campo_Relativo"])
    indices_sim = detectar_picos(sync["Sim_Relativo"])
    fechas_obs = [pd.Timestamp(sync.loc[i, "Fecha"]) for i in indices_obs]
    fechas_sim = [pd.Timestamp(sync.loc[i, "Fecha"]) for i in indices_sim]
    tolerancia = max(1, int(round(sync["Dias_Intervalo"].median())))

    primer_flujo_obs = primera_fecha(sync["Fecha"], sync["Campo_Relativo"], UMBRAL_EVENTO)
    primer_flujo_sim = primera_fecha(sync["Fecha"], sync["Sim_Relativo"], UMBRAL_EVENTO)
    primer_pico_obs = fechas_obs[0] if fechas_obs else pd.NaT

    # La frecuencia y el F1 de picos se evalúan sobre flujos Event-to-Event.
    # Sin embargo, el desfase del PRIMER pico debe conservar la resolución
    # diaria del modelo. Usar la fecha final del intervalo para el pico
    # simulado puede producir artificialmente delta=0 cuando el máximo diario
    # ocurrió varios días antes del muestreo de campo.
    primer_pico_sim_intervalo = fechas_sim[0] if fechas_sim else pd.NaT
    inicio = pd.to_datetime(
        globales.get("fecha_inicio_ventana", pd.NaT),
        errors="coerce",
    )
    primer_pico_sim = (
        pd.Timestamp(inicio)
        if pd.notna(inicio)
        else primer_pico_sim_intervalo
    )

    control = globales.get("fecha_control", pd.NaT)
    limite = globales.get("fecha_limite", pd.NaT)
    alerta = primera_fecha(diario["Fecha"], diario["EMERREL"], float(globales.get("umbral_er", 0.005)))

    total_campo = float(campo[flujo_campo].sum())
    pec = np.nan
    if total_campo > 0 and pd.notna(control):
        pec = 100 * campo.loc[pd.to_datetime(campo[fecha_campo]) <= control, flujo_campo].sum() / total_campo

    resultado: dict[str, Any] = {
        "Sitio": SITIO,
        "N_intervalos": len(sync),
        "Intervalo_mediano_dias": float(sync["Dias_Intervalo"].median()),
        "Picos_observados": len(fechas_obs),
        "Picos_simulados": len(fechas_sim),
        "Razon_picos_sim_obs": len(fechas_sim) / len(fechas_obs) if fechas_obs else np.nan,
        "Tolerancia_picos_dias": tolerancia,
        "Pearson_flujos": pearson,
        "NSE_flujos": nse,
        "F1_intervalos": f1_i,
        "Fecha_primer_flujo_observado": primer_flujo_obs,
        "Fecha_primer_flujo_simulado": primer_flujo_sim,
        "Delta_primer_flujo_dias": (primer_flujo_sim - primer_flujo_obs).days if pd.notna(primer_flujo_obs) and pd.notna(primer_flujo_sim) else np.nan,
        "Fecha_primer_pico_observado": primer_pico_obs,
        "Fecha_primer_pico_simulado": primer_pico_sim,
        "Delta_primer_pico_dias": (primer_pico_sim - primer_pico_obs).days if pd.notna(primer_pico_obs) and pd.notna(primer_pico_sim) else np.nan,
        "Metodo_delta_primer_pico": "pico diario validado simulado - pico observado",
        "Fecha_primer_pico_simulado_intervalo": primer_pico_sim_intervalo,
        "Delta_primer_pico_intervalo_dias": (primer_pico_sim_intervalo - primer_pico_obs).days if pd.notna(primer_pico_obs) and pd.notna(primer_pico_sim_intervalo) else np.nan,
        "Fecha_inicio_termico": inicio,
        "Fecha_alerta": alerta,
        "Fecha_control": control,
        "Fecha_limite": limite,
        "PEC_control_pct": float(pec),
        "Lead_time_dias": (control - alerta).days if pd.notna(control) and pd.notna(alerta) else np.nan,
        "Ventana_600_800_dias": (limite - control).days if pd.notna(control) and pd.notna(limite) else np.nan,
        "Objetivo_control_CD": float(globales.get("dga_optimo", 600)),
        "Limite_control_CD": float(globales.get("dga_critico", 800)),
        "Campo_es_acumulado": CAMPO_ES_ACUMULADO,
    }
    resultado.update(aparear_picos(fechas_obs, fechas_sim, tolerancia))
    return resultado


def serializar(valor: Any) -> Any:
    if isinstance(valor, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(valor) else pd.Timestamp(valor).isoformat()
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        return None if pd.isna(valor) else float(valor)
    return valor


def guardar(base: Path, resultado: dict[str, Any], sync: pd.DataFrame, diario: pd.DataFrame, salida: str) -> None:
    carpeta = base / salida
    carpeta.mkdir(parents=True, exist_ok=True)
    limpio = {clave: serializar(valor) for clave, valor in resultado.items()}
    (carpeta / f"metricas_{ID_SITIO}.json").write_text(json.dumps(limpio, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([limpio]).to_csv(carpeta / f"metricas_{ID_SITIO}.csv", index=False, encoding="utf-8-sig")
    sync.to_csv(carpeta / f"event_to_event_{ID_SITIO}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    columnas = [c for c in ("Fecha", "EMERREL", "DG", "Primer_Pico_Habilitado") if c in diario.columns]
    diario[columnas].to_csv(carpeta / f"serie_diaria_{ID_SITIO}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--salida", default=SALIDA)
    parser.add_argument("--campo-acumulado", action="store_true", default=CAMPO_ES_ACUMULADO)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    globales = ejecutar_app(base, campo_acumulado=args.campo_acumulado)
    sync = sincronizar_eventos(globales["df"], globales["df_campo"], globales["col_fecha"], globales["col_plm2"])
    resultado = metricas(globales, sync)
    guardar(base, resultado, sync, globales["df"], args.salida)

    print(json.dumps({k: serializar(v) for k, v in resultado.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
