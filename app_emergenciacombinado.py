# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Azul.

La aplicación científica original se conserva en
``app_emergenciacombinado_core.py``. Este archivo la ejecuta sin alterar su
lógica y agrega, al final de la interfaz, una descarga Excel completa de los
resultados generados.
"""
from pathlib import Path

_CORE_APP = Path(__file__).with_name("app_emergenciacombinado_core.py")
exec(
    compile(_CORE_APP.read_text(encoding="utf-8"), str(_CORE_APP), "exec"),
    globals(),
)


def _fecha_excel(valor):
    """Devuelve una fecha legible o una celda vacía."""
    if valor is None or pd.isna(valor):
        return ""
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


# La descarga se muestra únicamente cuando el motor generó resultados.
if "df" in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    reporte_excel = io.BytesIO()

    with pd.ExcelWriter(
        reporte_excel,
        engine="xlsxwriter",
        datetime_format="dd/mm/yyyy",
        date_format="dd/mm/yyyy",
    ) as writer:
        # 1. Serie diaria completa del modelo.
        df.to_excel(writer, index=False, sheet_name="Resultados_Diarios")

        # 2. Resultados de validación integrados por cada intervalo real.
        if (
            "df_sincronizado" in globals()
            and isinstance(df_sincronizado, pd.DataFrame)
            and not df_sincronizado.empty
        ):
            df_sincronizado.to_excel(
                writer,
                index=False,
                sheet_name="Validacion_Intervalos",
            )

        # 3. Observaciones originales de campo.
        if (
            "df_campo" in globals()
            and isinstance(df_campo, pd.DataFrame)
            and not df_campo.empty
        ):
            df_campo.to_excel(
                writer,
                index=False,
                sheet_name="Observaciones_Campo",
            )

        # 4. Resumen operativo para la toma de decisiones.
        pd.DataFrame(
            {
                "Indicador": [
                    "Localidad",
                    "Fecha de generación",
                    "Inicio del conteo térmico",
                    "Fecha objetivo de control",
                    "Fecha límite de la ventana",
                    "TT acumulado actual (°Cd)",
                    "TT pronosticado +7 días (°Cd)",
                    "Estado operativo",
                ],
                "Valor": [
                    "Azul (Buenos Aires)",
                    pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                    _fecha_excel(globals().get("fecha_inicio_ventana")),
                    _fecha_excel(globals().get("fecha_control")),
                    _fecha_excel(globals().get("fecha_limite")),
                    globals().get("dga_hoy", 0.0),
                    globals().get("dga_7dias", 0.0),
                    globals().get("msg_estado", ""),
                ],
            }
        ).to_excel(writer, index=False, sheet_name="Resumen_Decision")

        # 5. Métricas de validación campo-modelo.
        pd.DataFrame(
            {
                "Metrica": [
                    "PEC (%)",
                    "Lag control (días)",
                    "Lead time (días)",
                    "Pearson de flujos",
                    "NSE de flujos",
                    "KGE de flujos",
                    "RMSE acumulado",
                    "R2 acumulado",
                    "CCC acumulado",
                    "Desfase T50 (días)",
                    "F1-Score de coincidencia",
                    "Exactitud global",
                    "Hits",
                    "Misses",
                    "Falsos positivos",
                    "Correctos negativos",
                    "Desfase del primer flujo (días)",
                ],
                "Valor": [
                    globals().get("pec", 0.0),
                    globals().get("peak_lag", 0),
                    globals().get("lead_time", 0),
                    globals().get("pearson_r", 0.0),
                    globals().get("nse_flujos", 0.0),
                    globals().get("kge_flujos", 0.0),
                    globals().get("rmse_acum", 0.0),
                    globals().get("r2_acum", 0.0),
                    globals().get("ccc_acum", 0.0),
                    globals().get("desfase_t50", 0),
                    globals().get("f1_score_coincidencia", 0.0),
                    globals().get("exactitud_global", 0.0),
                    globals().get("hits_val", 0),
                    globals().get("misses_val", 0),
                    globals().get("falsos_pos_val", 0),
                    globals().get("correctos_neg_val", 0),
                    globals().get("lag_inicio_dias", "N/A"),
                ],
            }
        ).to_excel(writer, index=False, sheet_name="Metricas_Validacion")

        # 6. Parámetros empleados en la corrida.
        pd.DataFrame(
            {
                "Parametro": [
                    "T base (°C)",
                    "T óptima (°C)",
                    "T crítica (°C)",
                    "Capacidad superficial W Max (mm)",
                    "Coeficiente Ke",
                    "Exponente Kr",
                    "Modulador térmico",
                    "Umbral de termoinhibición (°C)",
                    "Choque hídrico 3 días (mm)",
                    "Umbral del primer pico",
                    "Cobertura de rastrojo (%)",
                    "TT objetivo de control (°Cd)",
                    "TT límite de ventana (°Cd)",
                ],
                "Valor": [
                    globals().get("t_base_val", ""),
                    globals().get("t_opt_max", ""),
                    globals().get("t_critica", ""),
                    globals().get("w_max_val", ""),
                    globals().get("ke_val", ""),
                    globals().get("exponente_kr", ""),
                    globals().get("mod_termico", ""),
                    globals().get("umbral_termoinhibicion", ""),
                    globals().get("umbral_choque_hidrico", ""),
                    globals().get("UMBRAL_PRIMER_PICO", ""),
                    globals().get("cobertura_pct", ""),
                    globals().get("dga_optimo", ""),
                    globals().get("dga_critico", ""),
                ],
            }
        ).to_excel(writer, index=False, sheet_name="Parametros_Modelo")

        # Ajuste simple de anchos para facilitar la lectura inmediata.
        for nombre_hoja, hoja in writer.sheets.items():
            hoja.freeze_panes(1, 0)
            hoja.autofilter(0, 0, 0, max(0, hoja.dim_colmax))
            hoja.set_column(0, max(0, hoja.dim_colmax), 18)

    reporte_excel.seek(0)

    st.divider()
    st.subheader("📥 Descarga de resultados")
    st.caption(
        "El archivo incluye resultados diarios, validación por intervalos, "
        "observaciones de campo, métricas, resumen operativo y parámetros."
    )
    st.download_button(
        label="📊 Descargar resultados completos en Excel",
        data=reporte_excel.getvalue(),
        file_name="PREDWEEM_Azul_Resultados_Completos.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
        key="descarga_excel_resultados_final",
    )
