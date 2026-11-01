# Campañas de Azul: emergencia y clima

Son los datos de *Lolium multiflorum* de Azul con que se validó la alerta. La actualización automática no modifica estas
carpetas (`meteo_daily.csv`, en cambio, se reemplaza con la campaña nueva el 1 de enero de cada año).

## 2026

| Archivo | Contenido |
|---|---|
| `emergencia_azul_2026_original.xlsx` | Planilla aportada (FECHA, PLM2), sin cambios; copiada del gemelo `PREDWEEM/AZUL_digitaltween`. |
| `emergencia_azul_2026.csv` | Los mismos datos con días desde el conteo anterior, fracción de cada intervalo y emergencia acumulada (%). |
| `clima_azul_2026.csv` | Copia de `meteo_daily.csv` al cierre de la campaña: 274 días (2026-01-01 – 2026-10-01), 505 mm. |

Conteos: 11 fechas del 01/03/2026 (cero inicial) al 01/09/2026, cada 11 a 29 días;
8224 plántulas/m² en total. `PLM2` son las plántulas nuevas por m² en el intervalo que termina en esa fecha. La planilla no
identifica la localidad; la asignó el autor.

Clima: reanálisis **ERA5** (respaldo de ERA5-Land) en la grilla de Azul (−36,87; −59,89), no observaciones de estación
(columnas `FUENTE` y `TIPO`).

**Para sumar una campaña**, cree la carpeta `<año>/` con `emergencia_azul_<año>.csv` (al menos `Fecha` y `PLM2`) y su clima.
Con dos o más campañas se podrá usar un calendario propio de Azul para la curva acumulada, como en Tres Arroyos.

## Huellas SHA-256

| Archivo | SHA-256 |
|---|---|
| `2026/clima_azul_2026.csv` | `32ea11c98b04055b162b59b3ea379f0feba8e38e9520b825afa8e46b3d6b4110` |
| `2026/emergencia_azul_2026.csv` | `cd22d6bc0cf33e8e0b17cc4d6d60df4a93b041db93af460d5519a3ab7634a0ed` |
| `2026/emergencia_azul_2026_original.xlsx` | `c793fc43b4a20fe5a38c4b937836bc1111df34602dd2935f1b178ce1bee91d3d` |
