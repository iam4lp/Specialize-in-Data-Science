# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import datetime

st.set_page_config(page_title="Website Performance & Prediction", layout="wide", page_icon="📈")

# Başlık
st.title("📈 Website Performance Analysis")
st.markdown("Bu proje, web sitesi trafik verilerini görselleştirir ve gelecekteki trafik yoğunluğunu tahmin eder.")

# 1. Veri Yükleme (Analiz için)
@st.cache_data
def load_data():
    df = pd.read_csv("src/data-export.csv", header=1)
    df.columns = [
        'Channel', 'DateHour', 'Users', 'Sessions', 'EngagedSessions',
        'AvgEngagementTime', 'EngagedSessionsPerUser', 'EventsPerSession',
        'EngagementRate', 'EventCount'
    ]
    df['DateTime'] = pd.to_datetime(df['DateHour'], format='%Y%m%d%H')
    return df

try:
    df = load_data()
except:
    st.error("Veri seti yüklenemedi. Lütfen 'data-export.csv' dosyasını kontrol edin.")
    st.stop()

# 2. Model Yükleme (Tahmin için)
try:
    model = joblib.load('src/traffic_model.pkl')
except:
    model = None

# --- TAB YAPISI ---
tab1, tab2 = st.tabs(["📊 Analiz Panosu (Dashboard)", "🔮 Trafik Tahmini (Prediction)"])

with tab1:
    st.header("Site Performans Özeti")
    
    # KPI Kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Kullanıcı", f"{df['Users'].sum():,}")
    col2.metric("Toplam Oturum", f"{df['Sessions'].sum():,}")
    col3.metric("Ort. Etkileşim Süresi", f"{df['AvgEngagementTime'].mean():.1f} sn")
    col4.metric("Ort. Etkileşim Oranı", f"%{df['EngagementRate'].mean()*100:.1f}")
    
    st.divider()
    
    # Grafikler
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Trafik Kaynakları (Channels)")
        channel_data = df.groupby('Channel')['Sessions'].sum().reset_index()
        fig_pie = px.pie(channel_data, values='Sessions', names='Channel', title='Kanal Bazlı Oturum Dağılımı')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("Saatlik Trafik Yoğunluğu")
        df['Hour'] = df['DateTime'].dt.hour
        hourly_data = df.groupby('Hour')['Sessions'].mean().reset_index()
        fig_line = px.line(hourly_data, x='Hour', y='Sessions', title='Saatlere Göre Ortalama Oturum Sayısı', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Zaman İçinde Trafik Değişimi")
    # Günlük bazda toplama
    daily_trend = df.groupby(df['DateTime'].dt.date)['Sessions'].sum().reset_index()
    fig_trend = px.area(daily_trend, x='DateTime', y='Sessions', title='Günlük Toplam Oturum Sayısı')
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("Gelecek Trafik Tahmini")
    st.write("Belirli bir gün ve saatte, seçilen kanaldan ne kadar trafik geleceğini tahmin edin.")
    
    if model:
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        
        with col_inp1:
            input_channel = st.selectbox("Trafik Kanalı", df['Channel'].unique())
        
        with col_inp2:
            days = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
            input_day = st.selectbox("Gün Seçiniz", list(days.keys()), format_func=lambda x: days[x])
            
        with col_inp3:
            input_hour = st.slider("Saat Seçiniz", 0, 23, 12)
            
        if st.button("Trafik Tahmin Et"):
            # Tahmin için veri hazırlığı
            input_df = pd.DataFrame({
                'Channel': [input_channel],
                'Hour': [input_hour],
                'DayOfWeek': [input_day]
            })
            
            prediction = model.predict(input_df)[0]
            
            st.success(f"Tahmini Oturum Sayısı (Sessions): **{int(prediction)}**")
            
            # Bağlam bilgisi
            avg_val = df[(df['Channel'] == input_channel) & (df['Hour'] == input_hour)]['Sessions'].mean()
            if not pd.isna(avg_val):
                st.info(f"Geçmiş verilerde bu saat ve kanal için ortalama: {int(avg_val)}")
    else:
        st.warning("Model dosyası ('traffic_model.pkl') bulunamadı. Lütfen önce modeli eğitin.")