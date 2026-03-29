import requests
import pandas as pd
import sys
import os

# Coordenadas de Pergamino
LAT = -33.8895
LON = -60.5736
ARCHIVO_CSV = 'meteo_daily.csv'

def actualizar_pronostico():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "forecast_days": 7
    }
    
    print("Consultando a Open-Meteo...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error en la API: {response.text}")
        sys.exit(1)
        
    data = response.json()
    
    # DataFrame con los nuevos 7 días
    df_nuevo = pd.DataFrame({
        'Fecha': data['daily']['time'],
        'TMAX': data['daily']['temperature_2m_max'],
        'TMIN': data['daily']['temperature_2m_min'],
        'Prec': data['daily']['precipitation_sum']
    })
    
    if df_nuevo.isnull().values.any():
        print("ERROR: La API devolvió datos incompletos o vacíos.")
        sys.exit(1)

    # Lógica para integrar con el historial
    if os.path.exists(ARCHIVO_CSV):
        print(f"Leyendo historial desde {ARCHIVO_CSV}...")
        df_historico = pd.read_csv(ARCHIVO_CSV)
        
        # Opcional: Si quieres forzar una limpieza estricta de cualquier dato basura 
        # que haya quedado del 27/03 en adelante antes de pegar el nuevo pronóstico,
        # descomenta la siguiente línea:
        # df_historico = df_historico[df_historico['Fecha'] < '2026-03-27']

        # Combinamos el historial con los datos nuevos
        df_final = pd.concat([df_historico, df_nuevo], ignore_index=True)
        
        # Eliminamos duplicados basados en la 'Fecha', conservando el 'last' (el pronóstico más reciente)
        # Esto asegura que el pasado queda intacto, pero los días futuros se actualizan
        df_final = df_final.drop_duplicates(subset=['Fecha'], keep='last')
        
        # Ordenamos cronológicamente por las dudas
        df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
    else:
        print(f"No se encontró {ARCHIVO_CSV}, creando uno nuevo...")
        df_final = df_nuevo

    # Guardar el archivo definitivo
    df_final.to_csv(ARCHIVO_CSV, index=False)
    print("Archivo actualizado exitosamente. Últimos 10 registros:")
    print(df_final.tail(10))

if __name__ == "__main__":
    actualizar_pronostico()
