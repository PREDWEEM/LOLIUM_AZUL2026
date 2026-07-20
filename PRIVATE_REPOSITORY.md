# Preparación para repositorio privado — PREDWEEM Azul

Este documento describe la verificación necesaria antes y después de cambiar la visibilidad de `PREDWEEM/LOLIUM_AZUL2026` a privada.

## Cambios técnicos incorporados

- La aplicación utiliza los datos meteorológicos incluidos en el checkout local.
- Se elimina la dependencia de `raw.githubusercontent.com/PREDWEEM/LOLIUM_AZUL2026`.
- Los pesos ANN y el clasificador deben existir realmente; la aplicación no genera modelos aleatorios de reemplazo.
- El logo local es opcional: si `logo.png` no está disponible, la interfaz muestra una identificación textual de PREDWEEM.
- Un workflow de preflight verifica recursos, sintaxis y ausencia de dependencias públicas internas.

## Antes de cambiar la visibilidad

1. Autorizar a Streamlit Community Cloud para acceder a repositorios privados de la cuenta u organización `PREDWEEM`.
2. Confirmar en Streamlit que la app utiliza:
   - repositorio: `PREDWEEM/LOLIUM_AZUL2026`;
   - rama: `main`;
   - archivo principal: `app_emergenciacombinado.py`.
3. Verificar en GitHub Actions que el workflow meteorológico programado esté habilitado.
4. Revisar que los secretos utilizados por Actions continúen configurados en `Settings → Secrets and variables → Actions`.
5. Ejecutar el workflow `Verificar despliegue privado` y confirmar que finalice correctamente.

## Cambio de visibilidad

En GitHub:

`Settings → General → Danger Zone → Change repository visibility → Make private`

## Después de privatizar

1. Ejecutar manualmente el workflow meteorológico existente mediante `workflow_dispatch`.
2. Confirmar que `meteo_daily.csv` sea actualizado y que el bot pueda crear el commit correspondiente.
3. Abrir la aplicación Streamlit y verificar:
   - carga del archivo meteorológico local;
   - carga de los pesos ANN y del clasificador;
   - funcionamiento del panel de Azul;
   - ausencia de errores de acceso a GitHub Raw;
   - presentación correcta de la identificación PREDWEEM.
4. Esperar al menos una ejecución programada para confirmar que el cron continúa funcionando.

## Actualización meteorológica

La privatización no impide que GitHub Actions consulte APIs meteorológicas públicas ni que escriba en el mismo repositorio mediante `GITHUB_TOKEN`, siempre que el workflow conserve el permiso `contents: write` y Actions esté habilitado.

## Próxima etapa de seguridad

La privatización protege las mejoras futuras, pero los archivos previamente públicos deben considerarse potencialmente copiados. La arquitectura definitiva recomendada consiste en trasladar pesos, clasificador y lógica científica a una API o repositorio privado central, dejando las aplicaciones como clientes del servicio.
