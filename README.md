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
ventana de control post-emergente (600–800 °Cd desde el inicio). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
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

La barra lateral conserva solo los parámetros que usa la alerta (temperaturas base, óptima y crítica, y objetivos 600 y
800 °Cd); los del motor PREDWEEM quedan en sus valores por defecto (cobertura de rastrojo 10 %).

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

## Aviso de flujos tardíos

Del 1 de mayo al 30 de junio la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña. El aviso se da desde 20 %; sin datos del lote, desde 25 % (`tardia.umbral_tramos`).
En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas por semana importan para decidir la fecha de siembra.
Es la misma alerta recalibrada por nivel de información (`tardia` en `alerta_lolmu_params.json`). Aparece en un recuadro
debajo de la alerta, en el **semáforo semanal** del gráfico (fila «Flujos tardíos»: violeta fuerte = flujo tardío probable;
la fila «Grandes flujos» repite el nivel de cada semana) y en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 130 semanas de mayo–junio, 42 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima (aviso desde 25 %) | 0,57 | 67 % | 68 % |
| + primeras plántulas | 0,79 | 90 % | 62 % |
| + conteos semanales | 0,87 | 86 % | 51 % |

Cada campaña se predijo con el ajuste sin ella. Sin datos del lote el aviso casi no distingue semanas: conviene cargar la
fecha de primeras plántulas. En otras seis localidades (58 intervalos de mayo–junio, 10 con ≥ 2 % por semana) la AUC fue
0,85 con primeras plántulas y 0,89 con conteos, con 90 % de detección; allí la probabilidad tiende a ser alta (30 % contra
17 % observado).

**Del 1 de julio al 1 de octubre no se muestra probabilidad** (`tardia.recorrida`): en Bordenave (80 semanas, 11 con
flujo tardío) no distinguió las semanas con emergencia (AUC 0,41–0,50), y bajar el umbral solo marcaba más semanas al
azar. En su lugar, la app recomienda **recorrer los lotes cada 2 semanas**, sobre todo donde suele haber emergencia
invernal (sudoeste bonaerense), antes de definir la fecha de siembra o un control. En el semáforo esas semanas aparecen
en gris.

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
