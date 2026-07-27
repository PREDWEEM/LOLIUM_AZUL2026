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

## Preparación para repositorio privado

La aplicación utiliza los recursos meteorológicos y científicos incluidos en el checkout local y no debe depender de URLs públicas del propio repositorio. Antes de cambiar la visibilidad, revise la guía [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md) y ejecute el workflow **Verificar despliegue privado**.

La actualización meteorológica programada puede continuar funcionando en un repositorio privado mediante GitHub Actions, siempre que Actions permanezca habilitado y el workflow conserve los permisos necesarios para actualizar `meteo_daily.csv`.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
