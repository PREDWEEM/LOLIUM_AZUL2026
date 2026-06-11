
import requests
import pandas as pd
import sys
import os

# Coordenadas específicas de AZUL, Provincia de Buenos Aires
LAT = -36.87
LON = -59.89
ARCHIVO_CSV = 'meteo_daily.csv'

def actualizar_pronostico():
    url = "https://api.open-meteo.com/v1/forecast"
    
    # ESTRATEGIA INTEGRAL: Ventana híbrida móvil.
    # Descarga los 7 días pasados consolidados por reanálisis y los 7 días de pronóstico técnico.
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "past_days": 7,
        "forecast_days": 7
    }
    
    print("Consultando a Open-Meteo para Azul (Ventana Híbrida: -7d a +7d)...")
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Error en la API: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Error de conexión con la API: {e}")
        sys.exit(1)
        
    data = response.json()
    
    # DataFrame con el bloque de 14 días móviles de Azul
    df_nuevo = pd.DataFrame({
        'Fecha': data['daily']['time'],
        'TMAX': data['daily']['temperature_2m_max'],
        'TMIN': data['daily']['temperature_2m_min'],
        'Prec': data['daily']['precipitation_sum']
    })
    
    # Forzar el parseo a datetime para evitar fallos por discrepancias de formato string
    df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'])
    
    if df_nuevo.isnull().values.any():
        print("ADVERTENCIA: Datos incompletos detectados para Azul. Aplicando interpolación forward-fill.")
        df_nuevo = df_nuevo.ffill()

    # Integración consistente con el archivo histórico local
    if os.path.exists(ARCHIVO_CSV):
        print(f"Leyendo historial desde {ARCHIVO_CSV}...")
        df_historico = pd.read_csv(ARCHIVO_CSV)
        df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
        
        # Combinamos la base histórica con el nuevo bloque temporal de la API
        df_final = pd.concat([df_historico, df_nuevo], ignore_index=True)
        
        # PROCESO DE CORRECCIÓN:
        # Al conservar el 'last', los 7 días pasados reales asimilados por Open-Meteo
        # eliminan y sobreescriben de forma automática los pronósticos antiguos guardados.
        df_final = df_final.drop_duplicates(subset=['Fecha'], keep='last')
        df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
    else:
        print(f"No se encontró {ARCHIVO_CSV}, creando un registro nuevo para Azul...")
        df_final = df_nuevo

    # Guardar manteniendo consistencia de formato de fecha ISO estricto (YYYY-MM-DD)
    df_final['Fecha'] = df_final['Fecha'].dt.strftime('%Y-%m-%d')
    df_final.to_csv(ARCHIVO_CSV, index=False)
    
    print("Base meteorológica de Azul sincronizada y purgada con éxito. Últimos 10 registros:")
    print(df_final.tail(10))

if __name__ == "__main__":
    actualizar_pronostico()
