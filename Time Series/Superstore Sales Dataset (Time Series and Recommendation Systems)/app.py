import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Superstore Pro AI", layout="wide")
st.title("🚀 Superstore AI: Profesyonel Karar Destek")
st.caption("AutoML & Grid Search Engine | v3.1 Visual Update")

DATA_FILE = "src/train.csv"
MODEL_FILE = "src/superstore_models.pkl"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    return df.groupby(['Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().reset_index()

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_FILE): return None
    return joblib.load(MODEL_FILE)

df_monthly = load_data()
models = load_models()

if df_monthly is None or models is None:
    st.error("Model dosyası eksik! Lütfen önce 'train_model.py' çalıştırın.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Kontrol Paneli")
    selected_cat = st.selectbox("Kategori:", list(models.keys()))
    steps = st.slider("Tahmin Süresi (Ay):", 3, 24, 12)
    
    st.divider()
    
    # Model Teknik Detayları
    model_data = models[selected_cat]
    cfg = model_data['config']
    
    st.info("🧠 **Aktif Model Parametreleri:**")
    st.json(cfg)
    st.caption(f"Veri Tipi: {'Logaritmik (Volatil)' if cfg['log'] else 'Lineer (Stabil)'}")

with col2:
    # Veriyi Hazırla
    cat_data = df_monthly[df_monthly['Category'] == selected_cat].set_index('Order Date')['Sales']
    
    # Tahmin Yap
    model = model_data['model']
    forecast_raw = model.forecast(steps)
    
    # Log ise geri çevir
    if cfg['log']:
        forecast = np.expm1(forecast_raw)
    else:
        forecast = forecast_raw
        
    # --- GRAFİK (Gelişmiş Görselleştirme) ---
    fig = go.Figure()
    
    # 1. Parça: GERÇEK VERİ (Gri)
    fig.add_trace(go.Scatter(
        x=cat_data.index, y=cat_data.values, 
        name='Gerçekleşen', 
        line=dict(color='gray', width=2)
    ))
    
    # 2. Parça: KÖPRÜ/GEÇİŞ (Turuncu - Kesikli)
    # Son gerçek veriden, ilk tahmin verisine uzanan çizgi
    last_real_date = cat_data.index[-1]
    last_real_val = cat_data.iloc[-1]
    first_pred_date = forecast.index[0]
    first_pred_val = forecast.iloc[0]
    
    fig.add_trace(go.Scatter(
        x=[last_real_date, first_pred_date], 
        y=[last_real_val, first_pred_val], 
        name='Geçiş', 
        mode='lines',
        line=dict(color='#FFA500', width=2, dash='dash'), # Turuncu ve Kesikli
        showlegend=False # Efsanede (Legend) kalabalık yapmasın
    ))
    
    # 3. Parça: AI TAHMİN (Mavi - Kalın)
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values, 
        name='AI Tahmin', 
        line=dict(color='#0068C9', width=4)
    ))
    
    fig.update_layout(
        title=f"📈 {selected_cat} Satış Trendi Analizi",
        xaxis_title="Tarih",
        yaxis_title="Satış Tutarı ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- METRİKLER ---
    st.subheader("📢 Yapay Zeka Tavsiyesi")
    
    last_val = cat_data.iloc[-1]
    next_val = forecast.iloc[0]
    change_pct = ((next_val - last_val) / last_val) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Son Gerçekleşen", f"${last_val:,.0f}")
    c2.metric("Gelecek Ay Tahmini", f"${next_val:,.0f}")
    c3.metric("Beklenen Değişim", f"%{change_pct:.1f}", delta_color="normal")
    
    st.divider()
    
    # --- KARAR MEKANİZMASI ---
    if change_pct > 10:
        st.success(f"🚀 **BÜYÜME ALARMI (%{change_pct:.1f})**")
        st.write(f"**{selected_cat}** kategorisinde ciddi talep artışı bekleniyor. Stokları doldurun.")
        
    elif change_pct < -10:
        st.error(f"📉 **DÜŞÜŞ UYARISI (%{change_pct:.1f})**")
        st.write(f"**{selected_cat}** kategorisinde düşüş öngörülüyor. Kampanya planlayın.")
        
    else:
        st.info(f"⚖️ **PİYASA STABİL (%{change_pct:.1f})**")
        st.write(f"**{selected_cat}** dengeli seyrediyor. Mevcut stratejiyi koruyun.")