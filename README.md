# PREDWEEM — Lolium Azul 2026

Repositorio correspondiente a la implementación de **PREDWEEM** para la predicción de la emergencia y la dinámica fenológica de *Lolium multiflorum* en Azul, provincia de Buenos Aires, Argentina.

> **Propiedad intelectual**  
> Copyright © 2026 Guillermo R. Chantre / PREDWEEM.  
> Todos los derechos reservados.
>
> Este repositorio constituye software propietario. Su disponibilidad pública no concede autorización para utilizar, copiar, modificar, redistribuir, sublicenciar, realizar ingeniería inversa ni explotar comercialmente el código, los modelos, los parámetros, los pesos neuronales, la documentación o los datos incluidos.
>
> Consulte el aviso completo en [COPYRIGHT.md](COPYRIGHT.md).

## Finalidad

PREDWEEM es una herramienta de apoyo a la toma de decisiones agronómicas basada en la integración de datos meteorológicos, modelos predictivos y filtros ecofisiológicos para anticipar los flujos de emergencia de raigrás anual.

La implementación de este repositorio está orientada a **Azul** y debe utilizarse considerando el dominio geográfico, climático y agronómico para el cual fue configurada, así como su estado específico de validación.

## Sin motor PREDWEEM (noviembre de 2026)

La página ya no ejecuta el motor PREDWEEM anterior (red neuronal y balance hídrico). Muestra solo la **alerta de grandes
flujos** del paquete `lolium-comun` con el clima diario del repositorio (`meteo_daily.csv`): qué hacer esta semana,
probabilidad de gran flujo, ventana del graminicida de barbecho, residual a la siembra, flujos tardíos y descarga en Excel.
El inicio de la emergencia estimado por el motor (su primer pico) erró 12–14 días en promedio en 31 campañas-sitio, no
mejor que una fecha fija, y la alerta ya no lo usaba: el reloj térmico parte de la fecha de primeras plántulas. El encabezado,
la barra lateral, la lectura del clima y la descarga vienen de `lolium_comun.pagina`. Los archivos del motor (`*_core.py`,
modelos `.npy`/`.pkl` y sus parches) quedan en el repositorio solo como referencia.

## Estrategia meteorológica

La serie operativa `meteo_daily.csv` utiliza las coordenadas de Azul `-36.87, -59.89` y separa explícitamente los datos según su origen y función:

- **ERA5-Land:** reanálisis histórico consolidado, utilizado hasta el último día que la API entrega completo.
- **ERA5:** respaldo explícito si ERA5-Land no presenta una ventana reciente válida.
- **ECMWF IFS histórico:** completa automáticamente los días posteriores al último reanálisis completo.
- **ECMWF IFS HRES:** pronóstico operativo desde el día actual hasta siete días posteriores.

Antes de descargar el histórico completo, el actualizador sondea una ventana reciente y detecta el último día con Tmax, Tmin y precipitación válidas. De esta manera, una demora transitoria de ERA5-Land no interrumpe el workflow: los días todavía incompletos pasan al bloque ECMWF IFS sin ser rellenados artificialmente.

El archivo conserva las columnas requeridas por el modelo (`Fecha`, `TMAX`, `TMIN`, `Prec`) y añade:

- `FUENTE`: producto meteorológico utilizado.
- `TIPO`: `REANALISIS`, `REANALISIS_FALLBACK`, `HISTORICO_MODELO` o `PRONOSTICO`.
- `FECHA_EMISION`: fecha y hora local de generación del bloque meteorológico.

La precipitación faltante **no se reemplaza por cero ni se arrastra desde el día anterior**. Si la API devuelve fechas discontinuas, duplicados, precipitación negativa o una temperatura máxima inferior a la mínima dentro de un bloque que debería estar completo, la actualización falla y conserva intacto el archivo operativo anterior.

La actualización automática se ejecuta diariamente a las **07:30** y **15:30**, hora de Argentina, y también puede iniciarse manualmente mediante GitHub Actions.

## Alerta de flushes

La página muestra la **alerta de flushes** de *Lolium multiflorum*: la probabilidad de que en los próximos 7 días emerja
≥ 5 % de la campaña, con dos niveles (*Atención* desde 15 %, *Alerta* desde 25 %), la emergencia acumulada esperada y la
ventana del graminicida de barbecho (600–800 °Cd, raigrás de 4 a 6 macollos desde el inicio). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
`visualizacion_alerta_lolmu.py`) es el calibrado con 17 campañas de Bordenave y combina, con la información disponible a la
fecha de emisión: el calendario medio de emergencia de LOLMU; la respuesta a las lluvias ya registradas; opcionalmente, la
fecha en que se vieron las primeras plántulas del lote (que además ancla el reloj térmico; sin ese dato la ventana de
control no se calcula); y, opcionalmente, conteos de plántulas (tabla o planilla Excel/CSV).

**Validación en Azul.** Sin reajustar el modelo, con los conteos de Azul 2026 (`data/campanias/2026/`; 10 intervalos de 11 a
29 días entre el 01/03 y el 01/09, 4 con flush) y el clima de este repositorio: AUC 1,00 en los tres niveles de información;
los 4 intervalos con flush quedaron en *Alerta* y ninguno de los 6 sin flush. El gemelo digital de Azul no pudo dar
porcentajes de emergencia durante 2026 (su referencia es la propia campaña 2026, disponible recién al cerrarla); su red sin
calibrar dio AUC 0,62. Límites: una sola campaña, intervalos largos (un resultado mucho menos exigente que una validación
semanal) y clima de reanálisis ERA5, no de estación. La curva acumulada es el calendario de Bordenave (error medio de 6 puntos
en Azul 2026).

La barra lateral tiene solo los parámetros que usa la alerta (temperaturas base, óptima y crítica, y objetivos 600 y
800 °Cd).

## Campaña vigente y actualización automática

La campaña meteorológica va del **1 de enero al 1 de octubre** de cada año (`_campania_vigente()` en `actualizar_clima.py`;
las variables `CAMPANIA_INICIO` y `CAMPANIA_FIN` la fijan a mano). Después del 1 de octubre la serie queda cerrada; el
**1 de enero siguiente** la carga se reinicia sola. En los primeros 21 días se admite no tener todavía reanálisis del año: ECMWF
IFS histórico cubre desde el 1 de enero. `test_actualizar_clima.py` prueba el reinicio.

## Preparación para repositorio privado

La aplicación utiliza los recursos meteorológicos y científicos incluidos en el checkout local y no debe depender de URLs públicas del propio repositorio. Antes de cambiar la visibilidad, revise la guía [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md) y ejecute el workflow **Verificar despliegue privado**.

La actualización meteorológica programada puede continuar funcionando en un repositorio privado mediante GitHub Actions, siempre que Actions permanezca habilitado y el workflow conserve los permisos necesarios para actualizar `meteo_daily.csv`.

**Inicio de la curva.** Cuando se informa la fecha de las primeras plántulas (o hay un primer conteo con plántulas), la curva
del calendario vale 0 antes de esa fecha, arranca ese día en la fracción típica ya emergida al ver las primeras plantas
(~18 %) y desde ahí sigue el calendario reescalado. En Azul 2026 el error medio de la curva desde las primeras plántulas
fue 11,0 puntos, contra 5,3 con el calendario por fecha (una sola campaña): se prefirió que la curva no suba antes del inicio
informado.

## Señal de lluvia (actualización de noviembre de 2026)

La señal de lluvia usa ahora el **modelo de pulsos ajustado con las 17 campañas de *Lolium multiflorum* de Bordenave**.
El tamaño de cada pulso depende del agua que entró al suelo, de cuán seco estaba y de la época del año (máximo para
lluvias de fines de mayo); la mitad de la respuesta emerge dentro de 114 °Cd de tiempo térmico de suelo y el 90 %
dentro de 252 °Cd. Con el nuevo núcleo se reajustaron los modelos logísticos, los percentiles de referencia y la
recalibración de flujos tardíos (`alerta_lolmu_params.json`). El balance de agua y la temperatura de suelo no cambian.

Validación con cada campaña de Bordenave predicha con el ajuste sin ella (incluido el núcleo de lluvia): AUC 0,77 solo
con clima, 0,83 con primeras plántulas y 0,87 con conteos. En las ocho localidades de validación, sin reajustar: 0,89,
0,92 y 0,93 (calendario solo, 0,89). En Azul: 1,00, 1,00 y 0,96 (antes 1,00, 1,00 y 1,00); las tablas de
validación de este sitio que figuran más arriba corresponden a la versión anterior.

## Flujos tardíos

Del **1 de agosto al 1 de octubre** la app recomienda **recorrer los lotes cada 2 semanas** para definir uso de graminicida
postemergente (`tardia.recorrida`): en esa época la probabilidad de flujo tardío no distinguió las semanas con emergencia
(AUC 0,41–0,50) y bajar el umbral solo marcaba más semanas al azar. Del 1/05 al 31/07 la emergencia que sigue a la siembra
se evalúa con el **riesgo para el residual** (ver más abajo), por eso la app ya no muestra la probabilidad de flujos tardíos
de mayo–junio (`tardia` sigue en los parámetros, sin uso en la pantalla).

## Qué hacer esta semana

Arriba de la alerta, un recuadro resume las acciones de la semana según la fecha y los datos cargados: recorrer el lote ante
un gran flujo probable o hasta ver las primeras plántulas; el momento del **graminicida de barbecho**; si conviene un
**residual** a la siembra prevista y si hay lluvia para activarlo; y la recorrida de flujos tardíos (1/08–1/10). El mismo resumen va en el Excel.

- **Graminicida de barbecho:** la ventana de 600–800 °Cd desde las primeras plántulas corresponde al raigrás de **4 a 6
  macollos**, momento óptimo para intervenir con un graminicida de barbecho.
- **Activación del residual:** suma la lluvia registrada y pronosticada en los 7 días desde la siembra; con 10 mm o más la
  activación es probable (el mínimo exacto depende del producto).
- **Flujos tardíos:** solo del 1/08 al 1/10 (recorrida cada 2 semanas); antes, la emergencia posterior a la siembra se
  evalúa con el riesgo para el residual.

## Residual a la siembra (guía de riesgo)

Debajo de la alerta, un recuadro siempre visible muestra, para la **fecha de siembra prevista** (1/05–31/07; si no se
carga, el 1/06 como referencia), el riesgo de nacimientos en la **ventana de acción del residual: de 10 días antes a 45 días
después de la siembra**, el período de acción promedio de un herbicida residual. Tiene tres tarjetas: riesgo, nacimientos en
la ventana (mediana de las campañas, en plantas/m² y en % de la campaña) y recomendación. Se calcula con las 34 campañas de la región, como referencia, porque Azul tiene una sola campaña; se muestra también el valor de esa campaña (`residual` en `alerta_lolmu_params.json`). Las plantas/m² usan la densidad de cada campaña con conteos; las campañas que solo
tienen la curva relativa usan la densidad de referencia del sitio (8224 pl/m² por campaña, mediana de las
campañas con conteos, aplicada a la distribución regional):

| Siembra | Nacimientos de −10 a +45 días (mediana) | Riesgo para el residual | Campañas del sitio |
|---|---|---|---|
| 1/05 | 1294 pl/m² (15,7 %) | Muy alto | 2026: 2528 pl/m² |
| 15/05 | 639 pl/m² (7,8 %) | Muy alto | 2026: 1560 pl/m² |
| 1/06 | 273 pl/m² (3,3 %) | Muy alto | 2026: 441 pl/m² |
| 15/06 | 213 pl/m² (2,6 %) | Muy alto | 2026: 605 pl/m² |
| 1/07 | 138 pl/m² (1,7 %) | Muy alto | 2026: 503 pl/m² |

El nivel de riesgo surge de las plantas/m² que nacieron en la ventana en una campaña típica (mediana). Se usan plantas
y no el % de la campaña porque en sitios con mucha densidad una fracción chica son muchas plantas (en San Pedro, 0,8 % de
la campaña fueron 29–42 pl/m²):

| Riesgo | Nacimientos en la ventana | Recomendación |
|---|---|---|
| 🔴 Muy alto | ≥ 50 pl/m² | Usar residual |
| 🟠 Alto | 20–50 pl/m² | Residual recomendado |
| 🟡 Medio | 5–20 pl/m² | Según infestación: residual si el lote tiene alta infestación o antecedentes de raigrás |
| 🟢 Bajo | < 5 pl/m² | Residual solo en lotes muy infestados |

Sin otro dato, las densidades son las registradas en los ensayos del sitio. El usuario puede indicar la **infestación de raigrás de su lote** (opcional): baja (unas 200 pl/m² en toda la campaña), media (unas 1.000) o alta (unas 4.000). Con ese dato las plantas/m² se calculan con la densidad del lote y, como la infestación ya está considerada, las recomendaciones de los niveles medio y bajo pasan a «residual opcional» y «no hace falta residual».

Debajo de las tarjetas se indica el valor de cada campaña del sitio y qué proporción de las campañas cayó en cada nivel. Es
un **riesgo histórico, no una predicción de la campaña**: ni la fecha de primeras plántulas ni el clima anticiparon cuánta
emergencia queda después de la siembra.

En el gráfico, la segunda fila del **semáforo semanal** («Residual (siembra)») muestra, para cada semana del 1/05 al
31/07, el riesgo para el residual si se siembra el primer día de esa semana: rojo, alto o muy alto (≥ 20 pl/m²); amarillo, medio
(5–20 pl/m²); verde, bajo (< 5 pl/m²), con la infestación del lote si se indicó. Con fecha de siembra cargada, las semanas empiezan ese día, así la celda
de la siembra coincide con el recuadro del residual. La fecha de siembra prevista se marca
con una línea verde y una flecha con cartel.

## Control de ingreso

El acceso es **libre y gratuito, con registro**: se ingresa con una cuenta de Google y no hace falta aprobación. Se activa en
Streamlit Community Cloud → Settings → Secrets con `[auth]` (cliente OAuth de Google) y `[acceso]` con `activo = true` y
`registro_libre = true`; opcionalmente `[acceso.registro]` anota cada ingreso (email, nombre, app) en un formulario de
Google. Sin esos secretos la app queda abierta, sin ingreso. Detalle en `lolium_comun/acceso_predweem.py`.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**

## Fin de carga de cada campaña

La última fecha meteorológica de cada campaña es el **1 de octubre, inclusive** (en 2026, el 1/10/2026). El pronóstico se
recorta al alcanzar ese límite y desde el 2 de octubre no se consulta el pronóstico futuro. Las consultas históricas ERA5-Land,
ERA5 y ECMWF IFS quedan limitadas al 1 de octubre, permitiendo actualizar el reanálisis de los días pendientes sin ampliar la
campaña. La serie cerrada debe ser continua y no admite fechas posteriores al límite. El 1 de enero siguiente comienza la
campaña nueva.

## Módulo común (LOLIUM-COMUN)

La alerta, su visualización y el control de acceso son comunes a las 8 apps de Lolium y están en el paquete
[PREDWEEM/LOLIUM-COMUN](https://github.com/PREDWEEM/LOLIUM-COMUN), fijado por versión en `requirements.txt`
(`lolium-comun @ git+https://github.com/PREDWEEM/LOLIUM-COMUN@<commit>  # v1.6.0`). En esta app, `alerta_lolmu.py`,
`visualizacion_alerta_lolmu.py` y `acceso_predweem.py` son archivos puente que cargan ese paquete con los parámetros de este
sitio (`alerta_lolmu_params.json`), que siguen acá. Los cambios a la alerta se hacen en LOLIUM-COMUN y llegan a la app al
actualizar la versión en `requirements.txt`.
