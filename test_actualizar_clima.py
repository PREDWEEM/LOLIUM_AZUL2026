import contextlib
import io
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import actualizar_clima as clima


def bloque(inicio, fin, tipo="REANALISIS_FALLBACK"):
    fechas = pd.date_range(inicio, fin)
    return pd.DataFrame({
        "Fecha": fechas, "TMAX": 20., "TMIN": 10., "Prec": 1.,
        "FUENTE": "ERA5", "TIPO": tipo, "FECHA_EMISION": "prueba",
    })


class RecuperacionMeteorologica(unittest.TestCase):
    def test_recupera_desde_primer_dia_pendiente(self):
        viejo = bloque("2026-01-01", "2026-08-11")
        self.assertEqual(clima._inicio_recuperacion(viejo, date(2026, 9, 5)), date(2026, 8, 12))

    def test_detecta_hueco_interior_y_dato_nulo(self):
        viejo = bloque("2026-01-01", "2026-09-04")
        viejo.loc[10, "Prec"] = float("nan")
        self.assertEqual(clima._inicio_recuperacion(viejo, date(2026, 9, 5)), date(2026, 1, 11))
        self.assertEqual(clima._inicio_recuperacion(viejo.drop(index=3), date(2026, 9, 5)), date(2026, 1, 4))

    def test_historial_completo_solo_refresca_cola(self):
        self.assertEqual(clima._inicio_recuperacion(bloque("2026-01-01", "2026-09-04"), date(2026, 9, 5)), date(2026, 9, 5))
        self.assertEqual(clima._inicio_recuperacion(bloque("2026-01-01", "2025-12-31"), date(2026, 9, 5)), clima.FECHA_INICIO)

    def test_integracion_archivo_estancado_y_escritura_atomica(self):
        for incompleto in (False, True):
            with self.subTest(incompleto=incompleto), tempfile.TemporaryDirectory() as tmp:
                destino = Path(tmp) / "meteo_daily.csv"
                bloque("2026-01-01", "2026-08-11").to_csv(destino, index=False)
                original = destino.read_bytes()
                consultas = []
                def descargar(inicio, fin, **kw):
                    consultas.append((inicio, fin))
                    df = bloque(inicio, fin, kw["tipo"])
                    df["FUENTE"] = kw["fuente"]
                    if incompleto:
                        df = df[df.Fecha != pd.Timestamp("2026-08-15")]
                    return df
                with patch.object(clima, "ARCHIVO_CSV", destino), patch.object(clima, "_ahora_local", return_value=datetime(2026, 9, 19, 12)), patch.object(clima, "_resolver_corte_reanalisis", return_value=(date(2026, 9, 11), "era5", "ERA5", "REANALISIS_FALLBACK")), patch.object(clima, "_descargar_historico_modelo", side_effect=descargar), patch.object(clima, "_descargar_pronostico", return_value=bloque("2026-09-19", "2026-09-26", "PRONOSTICO")), contextlib.redirect_stdout(io.StringIO()):
                    if incompleto:
                        with self.assertRaises(RuntimeError):
                            clima.actualizar_meteorologia()
                        self.assertEqual(destino.read_bytes(), original)
                    else:
                        salida = clima.actualizar_meteorologia()
                        self.assertEqual(len(salida), 269)
                        self.assertEqual(salida.Fecha.max(), "2026-09-26")
                    self.assertEqual(consultas[0][0], date(2026, 8, 12))


if __name__ == "__main__":
    unittest.main()
