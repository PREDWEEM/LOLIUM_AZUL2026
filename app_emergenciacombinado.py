# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM PRO v5.0 — "AZUL PLATINUM" 2026
# Edición: Guillermo R. Chantre
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path

# Nuevas librerías para UI moderna
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y THEME "MODERN SAAS"
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM Pro | Lolium 2026",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# Estilos CSS Avanzados (Tipografía Inter/Satoshi y Cards Modernas)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo principal */
    .stApp {
        background-color: #fcfcfd;
    }

    /* Estilo para el contenedor de métricas */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #f0f0f5;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }

    /* Alertas Personalizadas */
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .status-critical { background: #fff1f2; border-left-color: #e11d48; color: #9f1239; }
    .status-warning { background: #fffbeb; border-left-color: #d97706; color: #92400e; }
    .status-ok { background: #f0fdf4; border-left-color: #16a34a; color: #166534; }

    /* Esconder headers de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. LÓGICA TÉCNICA (SE MANTIENE INTEGRA)
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-36.78):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo_max=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0
    for i in range(1, n):
        kr = w[i-1] / w_max
        ke_dinamico = ke_suelo_max * kr
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - (et0[i] * ke_dinamico)))
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min, self.input_max = np.array([1, 0, -7, 0]), np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = [(self.LW @ np.tanh(self.IW.T @ x + self.bIW) + self.bLW) for x in Xn]
        emer = (np.tanh(np.array(emer).flatten()) + 1) / 2
        return np.diff(np.cumsum(emer), prepend=0), np.cumsum(emer)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except: return None, None

# ---------------------------------------------------------
# 3. SIDEBAR REESTRUCTURADO
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM-AZUL2026/main/logo.png", width=200)
    st.markdown("### 🛠️ Configuración del Modelo")
    
    with st.expander("🌡️ Parámetros Térmicos", expanded=False):
        t_base_val = st.number_input("T Base", 0.0, 10.0, 2.0)
        t_opt_max = st.number_input("T Óptima Max", 15.0, 35.0, 20.0)
        t_critica = st.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
        umbral_termoinhibicion = st.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0)

    with st.expander("💧 Parámetros Hídricos", expanded=False):
        w_max_val = st.number_input("Cap. Campo (mm)", 10.0, 100.0, 30.0)
        umbral_choque_hidrico = st.slider("Choque Hídrico 3d (mm)", 20, 100, 45)

    with st.expander("🎯 Estrategia de Control", expanded=True):
        umbral_er = st.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.30)
        dga_optimo = st.number_input("TT Control (°Cd)", 200, 1200, 600)
        dga_critico = st.number_input("Límite Ventana (°Cd)", 500, 1500, 800)
        residualidad = st.number_input("Residualidad (días)", 0, 60, 20)

    add_vertical_space(2)
    st.info("v5.0 Platinum - Desarrollado para el Sudeste Bonaerense.")

# ---------------------------------------------------------
# 4. CUERPO PRINCIPAL (DASHBOARD)
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

# Header Minimalista
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🌾 PREDWEEM | Lolium Azul")
    st.markdown("_Sistema de Apoyo a las Decisiones en Tiempo Real_")

with st.expander("📂 Gestión de Datos del Lote", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        archivo_meteo = st.file_uploader("Subir Clima (.csv / .xlsx)", type=["csv", "xlsx"])
    with c2:
        tipo_manejo = st.selectbox("Condición de Cobertura", 
            ["Alta Cobertura (SD)", "Media (SD Soja)", "Baja / Labranza"])
        # Mapeo de lógica
        mod_termico = 0.90 if "Alta" in tipo_manejo else (0.95 if "Media" in tipo_manejo else 1.0)
        ke_val = 0.25 if "Alta" in tipo_manejo else (0.50 if "Media" in tipo_manejo else 0.95)

# ---------------------------------------------------------
# 5. PROCESAMIENTO Y DASHBOARD
# ---------------------------------------------------------
df_raw = pd.read_csv(BASE/"meteo_daily.csv") if archivo_meteo is None else (pd.read_excel(archivo_meteo) if archivo_meteo.name.endswith(".xlsx") else pd.read_csv(archivo_meteo))

if df_raw is not None and modelo_ann is not None:
    # --- Motor de Cálculo (Optimizado) ---
    df = df_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Simulación Suelo
    df["TMAX_s"] = ((df["TMAX"] + df["TMIN"])/2) + (((df["TMAX"]-df["TMIN"])/2) * mod_termico)
    df["TMIN_s"] = ((df["TMAX"] + df["TMIN"])/2) - (((df["TMAX"]-df["TMIN"])/2) * mod_termico)
    
    # ANN & BHS
    emer_raw, _ = modelo_ann.predict(df[["Julian_days", "TMAX_s", "TMIN_s", "Prec"]].to_numpy(float))
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values)
    df["W_sup"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max_val, ke_val)
    
    # Factores Mecanísticos
    h_rel = df["W_sup"] / w_max_val
    f_hidrico = 1 / (1 + np.exp(-10 * (h_rel - 0.3)))
    df["EMERREL"] = emer_raw * f_hidrico
    df.loc[h_rel < 0.20, "EMERREL"] = 0.0
    df.loc[~(df['Prec'] >= w_max_val).cummax(), "EMERREL"] = 0.0 # Bloqueo hasta recarga
    
    # Escudo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"])/2
    df.loc[df["Tmedia"].rolling(10).mean() >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # --- MÉTRICAS PRINCIPALES ---
    idx_picos = df.index[df["EMERREL"] >= umbral_er].tolist()
    dga_hoy, status_msg, color_class = 0.0, "Esperando Pico", "status-warning"
    
    if idx_picos:
        f_inicio = df.loc[idx_picos[0], "Fecha"]
        dga_hoy = df[df["Fecha"] >= f_inicio]["DG"].sum()
        if dga_hoy >= dga_optimo:
            status_msg, color_class = f"🎯 MOMENTO CRÍTICO: Controlar ahora ({dga_hoy:.0f}°Cd)", "status-critical"
        else:
            status_msg, color_class = f"🌱 Ventana Activa: {dga_hoy:.0f}/{dga_optimo} °Cd acumulados", "status-ok"

    # UI: Fila de Kpis
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Emergencia Hoy", f"{df['EMERREL'].iloc[-1]*100:.1f}%", help="Tasa de emergencia diaria")
    m2.metric("TT Acumulado", f"{dga_hoy:.0f} °Cd", f"{dga_optimo - dga_hoy:.0f} p/ control", delta_color="inverse")
    m3.metric("Agua en Suelo", f"{df['W_sup'].iloc[-1]:.1f} mm", f"{h_rel.iloc[-1]*100:.0f}% Cap.")
    m4.metric("Predicción 7d", f"+{df['DG'].iloc[-10:-1].mean()*7:.0f} °Cd")
    style_metric_cards()

    # UI: Alerta Dinámica
    st.markdown(f'<div class="status-card {color_class}"><b>ESTADO DEL LOTE:</b> {status_msg}</div>', unsafe_allow_html=True)

    # TABS Modernas
    tab_mon, tab_suelo, tab_bio = st.tabs(["📊 Monitor de Decisión", "💧 Balance Hídrico", "🧪 Análisis Bio"])

    with tab_mon:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', name="Emergencia", line_color='#16a34a'))
        fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text="Alerta")
        fig.update_layout(template="plotly_white", hovermode="x unified", height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_suelo:
        fig_h = go.Figure()
        fig_h.add_bar(x=df["Fecha"], y=df["Prec"], name="Lluvia", marker_color="#93c5fd")
        fig_h.add_trace(go.Scatter(x=df["Fecha"], y=df["W_sup"], name="Agua Suelo", line_color="#0284c7"))
        fig_h.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_h, use_container_width=True)

    with tab_bio:
        # Mini curvas fisiológicas
        c_bio1, c_bio2 = st.columns(2)
        with c_bio1:
            st.caption("Respuesta Térmica (°Cd)")
            xt = np.linspace(0, 40, 100)
            yt = [calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica) for x in xt]
            st.line_chart(pd.DataFrame({'T': xt, 'TT': yt}).set_index('T'), height=200)

    # Exportación Pro
    st.sidebar.divider()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Simulada')
    st.sidebar.download_button("📥 Descargar Reporte PDF/Excel", output.getvalue(), "PREDWEEM_Report.xlsx", use_container_width=True)

else:
    st.warning("⚠️ Cargue un archivo de clima o asegúrese de que 'meteo_daily.csv' esté en la carpeta raíz.")
