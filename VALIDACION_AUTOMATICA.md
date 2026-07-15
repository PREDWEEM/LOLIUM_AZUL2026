# Validación automática PREDWEEM — Azul

Esta integración incorpora una página multipágina de Streamlit y un cálculo reproducible de métricas a partir de los archivos ya presentes en el repositorio:

- `app_emergenciacombinado.py`;
- `meteo_daily.csv`;
- `VALIDA (1).xlsx`;
- pesos ANN y modelo DTW.

## Archivos incorporados

```text
calcular_metricas_validacion.py
pages/2_Validacion.py
.github/workflows/calcular_metricas_validacion.yml
```

## Uso en Streamlit

Ejecutar la aplicación principal normalmente:

```bash
streamlit run app_emergenciacombinado.py
```

Streamlit detecta el directorio `pages/` y agrega la página **Validación** en la navegación lateral.

La página:

- ejecuta el modelo real de Azul con los parámetros predeterminados;
- utiliza automáticamente `meteo_daily.csv` y `VALIDA (1).xlsx`;
- integra la tasa diaria simulada entre fechas reales de monitoreo;
- incluye el primer intervalo observado;
- permite modificar el umbral de flujo significativo y la prominencia de los picos;
- presenta métricas, gráficos y descargas.

## Indicadores

- picos observados y simulados;
- razón de picos simulados/observados;
- hits, omisiones y falsos picos;
- F1 de picos;
- F1 de intervalos activos;
- Pearson y NSE de flujos Event-to-Event;
- desfase del primer flujo y del primer pico;
- fecha de inicio térmico;
- fecha de control a 600 °Cd;
- fecha límite a 800 °Cd;
- PEC al control;
- lead time;
- duración calendario de la ventana 600–800 °Cd.

## Ejecución sin interfaz

```bash
python calcular_metricas_validacion.py
```

Se generan:

```text
resultados_validacion/
├── metricas_azul.csv
├── metricas_azul.json
├── event_to_event_azul.csv
└── serie_diaria_azul.csv
```

## Datos de campo

La configuración inicial considera que `VALIDA (1).xlsx` contiene flujos de nuevas emergencias por intervalo. Cuando los valores sean acumulados, debe activarse la opción correspondiente en la página Streamlit o ejecutar:

```bash
python calcular_metricas_validacion.py --campo-acumulado
```

## Automatización

GitHub Actions verifica la sintaxis y recalcula las métricas cuando cambian el modelo, la meteorología, los pesos ANN o `VALIDA (1).xlsx`. En `main`, las salidas actualizadas se guardan automáticamente en `resultados_validacion/`.
