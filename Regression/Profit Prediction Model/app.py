# app.py
import streamlit as st
import pandas as pd
import joblib

# Sayfa Ayarları
st.set_page_config(page_title="Startup Profit Prediction", page_icon="💰")

st.title("💰 Startup Profit Prediction")
st.markdown("Bu uygulama, bir Startup şirketinin harcama kalemlerine göre tahmini yıllık kârını hesaplar.")

# 1. Modeli Yükle
try:
    model = joblib.load('src/profit_model.pkl')
except:
    st.error("Model dosyası bulunamadı. Lütfen önce 'train_model.py' dosyasını çalıştırın.")
    st.stop()

# 2. Kullanıcı Girdileri (Sidebar)
st.sidebar.header("Şirket Bilgileri")

rd_spend = st.sidebar.number_input("Ar-Ge Harcaması (R&D Spend)", min_value=0.0, value=70000.0, step=1000.0)
admin_spend = st.sidebar.number_input("Yönetim Giderleri (Administration)", min_value=0.0, value=100000.0, step=1000.0)
marketing_spend = st.sidebar.number_input("Pazarlama Harcaması (Marketing)", min_value=0.0, value=200000.0, step=1000.0)

state = st.sidebar.selectbox("Eyalet (State)", ['New York', 'California', 'Florida'])

# 3. Ana Ekranda Özet Gösterimi
col1, col2, col3 = st.columns(3)
col1.metric("Ar-Ge", f"${rd_spend:,.0f}")
col2.metric("Yönetim", f"${admin_spend:,.0f}")
col3.metric("Pazarlama", f"${marketing_spend:,.0f}")

st.write("---")

# 4. Tahmin Butonu
if st.button("Tahmini Kârı Hesapla (Predict Profit)"):
    # Girdileri modele uygun formata getir
    input_data = pd.DataFrame({
        'R&D Spend': [rd_spend],
        'Administration': [admin_spend],
        'Marketing Spend': [marketing_spend],
        'State': [state]
    })
    
    # Tahmin yap
    prediction = model.predict(input_data)[0]
    
    st.success(f"Tahmini Yıllık Kâr: **${prediction:,.2f}**")
    
    # Basit bir analiz mesajı
    if prediction > 150000:
        st.balloons()
        st.write("🚀 Harika bir performans! Bu kâr marjı oldukça yüksek.")
    elif prediction > 100000:
        st.write("✅ Başarılı bir Startup performansı.")
    else:
        st.warning("⚠️ Kâr marjı düşük görünüyor. Harcamaları gözden geçirmekte fayda var.")