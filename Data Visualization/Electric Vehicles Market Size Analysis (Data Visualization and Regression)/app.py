# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Sayfa Ayarları
st.set_page_config(page_title="EV Market Analysis", layout="wide")

st.title("🔋 Electric Vehicles Market Size Analysis")
st.markdown("Bu proje, elektrikli araç popülasyonunu analiz eder ve araç menzilini tahminleyen bir makine öğrenmesi modeli içerir.")

# 1. Veriyi Yükle (Cache mekanizması ile hızlandırıyoruz)
@st.cache_data
def load_data():
    df = pd.read_csv("src/Electric_Vehicle_Population_Data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Veri seti bulunamadı. Lütfen 'Electric_Vehicle_Population_Data.csv' dosyasını yükleyin.")
    st.stop()

# --- BÖLÜM 1: Pazar Analizi (Visualization) ---
st.header("1. Pazar Büyüklüğü ve Analizi")

# Analiz 1: Yıllara Göre Araç Sayısı
year_counts = df['Model Year'].value_counts().sort_index().reset_index()
year_counts.columns = ['Year', 'Count']
fig_year = px.bar(year_counts, x='Year', y='Count', title="Yıllara Göre Elektrikli Araç Sayısı")
st.plotly_chart(fig_year, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Analiz 2: En Popüler Markalar
    top_makes = df['Make'].value_counts().head(10).reset_index()
    top_makes.columns = ['Make', 'Count']
    fig_make = px.pie(top_makes, values='Count', names='Make', title="En Popüler 10 EV Markası")
    st.plotly_chart(fig_make, use_container_width=True)

with col2:
    # Analiz 3: Araç Tipi Dağılımı
    type_counts = df['Electric Vehicle Type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig_type = px.bar(type_counts, x='Type', y='Count', color='Type', title="EV Tipi Dağılımı (BEV vs PHEV)")
    st.plotly_chart(fig_type, use_container_width=True)

# --- BÖLÜM 2: Menzil Tahmin Modeli (Regression) ---
st.header("2. Elektrikli Menzil Tahmini (ML Model)")
st.write("Eğitilen modeli kullanarak bir aracın tahmini menzilini hesaplayın.")

# Modeli Yükle
try:
    model = joblib.load('src/ev_range_model.pkl')
    model_loaded = True
except:
    st.warning("Model dosyası ('ev_range_model.pkl') bulunamadı. Lütfen önce 'train_model.py' dosyasını çalıştırın.")
    model_loaded = False

if model_loaded:
    # Kullanıcı Girdileri
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        input_year = st.number_input("Model Yılı", min_value=2010, max_value=2025, value=2023)
    
    with col_input2:
        # Veri setindeki markaları seçenek olarak sunalım
        available_makes = sorted(df['Make'].unique())
        input_make = st.selectbox("Marka", available_makes, index=available_makes.index('TESLA') if 'TESLA' in available_makes else 0)
    
    with col_input3:
        # Araç tiplerini seçenek olarak sunalım
        available_types = df['Electric Vehicle Type'].unique()
        input_type = st.selectbox("Araç Tipi", available_types)
    
    # Tahmin Butonu
    if st.button("Menzili Tahmin Et"):
        # Girdiyi DataFrame'e çevir
        input_data = pd.DataFrame({
            'Model Year': [input_year],
            'Make': [input_make],
            'Electric Vehicle Type': [input_type]
        })
        
        # Tahmin
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Tahmini Menzil: **{prediction:.0f} mil**")
            
            # Kıyaslama için veri setinden benzer araçların ortalaması
            similar_cars = df[(df['Make'] == input_make) & (df['Electric Vehicle Type'] == input_type)]
            if not similar_cars.empty:
                avg_range = similar_cars[similar_cars['Electric Range'] > 0]['Electric Range'].mean()
                if pd.notna(avg_range):
                    st.info(f"Bilgi: Veri setindeki '{input_make}' markalı bu tip araçların ortalama menzili: {avg_range:.0f} mil.")
        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")