# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Sayfa Ayarları
st.set_page_config(page_title="Mobile Price Prediction", layout="wide")

st.title("📱 Mobile Price Classification")
st.markdown("Telefon özelliklerini girerek fiyat segmentini (0-3) tahmin edin.")

# 1. Model ve Scaler Yükleme
@st.cache_resource
def load_models():
    try:
        model = joblib.load('src/mobile_price_model.pkl')
        scaler = joblib.load('src/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_models()

if model is None:
    st.error("Model dosyaları bulunamadı. Lütfen önce 'train_model.py' dosyasını çalıştırın.")
    st.stop()

# 2. Kullanıcı Girdileri (Gruplandırılmış)
st.sidebar.header("Özellikleri Giriniz")

with st.expander("⚙️ Temel Donanım", expanded=True):
    col1, col2 = st.columns(2)
    ram = col1.number_input("RAM (MB)", 256, 8000, 2048)
    battery_power = col2.number_input("Pil Gücü (mAh)", 500, 6000, 2000)
    n_cores = col1.slider("Çekirdek Sayısı", 1, 8, 4)
    clock_speed = col2.slider("İşlemci Hızı (GHz)", 0.5, 3.0, 1.5)
    int_memory = st.number_input("Dahili Hafıza (GB)", 2, 256, 16)

with st.expander("📺 Ekran ve Tasarım"):
    col3, col4 = st.columns(2)
    px_height = col3.number_input("Piksel Yüksekliği", 0, 3000, 1000)
    px_width = col4.number_input("Piksel Genişliği", 0, 3000, 1000)
    sc_h = col3.number_input("Ekran Yüksekliği (cm)", 5, 25, 12)
    sc_w = col4.number_input("Ekran Genişliği (cm)", 0, 20, 5)
    m_dep = st.slider("Telefon Kalınlığı (cm)", 0.1, 1.0, 0.5)
    mobile_wt = st.slider("Ağırlık (gr)", 80, 300, 140)
    touch_screen = st.selectbox("Dokunmatik Ekran", [0, 1], format_func=lambda x: "Var" if x==1 else "Yok")

with st.expander("📸 Kamera"):
    pc = st.slider("Arka Kamera (MP)", 0, 20, 5)
    fc = st.slider("Ön Kamera (MP)", 0, 20, 2)

with st.expander("📡 Bağlantı Özellikleri"):
    col5, col6 = st.columns(2)
    blue = col5.checkbox("Bluetooth", value=True)
    wifi = col6.checkbox("WiFi", value=True)
    dual_sim = col5.checkbox("Çift SIM", value=True)
    four_g = col6.checkbox("4G", value=True)
    three_g = col5.checkbox("3G", value=True)

# 3. Tahmin İşlemi
if st.button("Fiyat Aralığını Tahmin Et"):
    # Girdileri listeye çevir (Sırası train verisiyle AYNI olmalı)
    # Sıralama: battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, 
    # n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi
    
    # Talk time sorulmadı, varsayılan atayalım veya ekleyelim. (Eksik kalmasın)
    talk_time = 10 # Varsayılan
    
    features = [
        battery_power, 
        1 if blue else 0, 
        clock_speed, 
        1 if dual_sim else 0, 
        fc, 
        1 if four_g else 0, 
        int_memory, 
        m_dep, 
        mobile_wt, 
        n_cores, 
        pc, 
        px_height, 
        px_width, 
        ram, 
        sc_h, 
        sc_w, 
        talk_time, 
        1 if three_g else 0, 
        1 if touch_screen==1 else 0, 
        1 if wifi else 0
    ]
    
    # Modele vermeden önce boyutlandırma ve scale
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    prediction = model.predict(features_scaled)[0]
    
    # Sonuç Gösterimi
    price_map = {
        0: "Düşük Maliyet (Low Cost)",
        1: "Orta Maliyet (Medium Cost)",
        2: "Yüksek Maliyet (High Cost)",
        3: "Çok Yüksek Maliyet (Very High Cost)"
    }
    
    st.success(f"Tahmin Edilen Fiyat Segmenti: **{price_map[prediction]}**")
    
    if prediction == 3:
        st.balloons()