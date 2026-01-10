import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Interactive Clustering", layout="wide")
st.title("📊 3D Customer Segmentation")
st.markdown("**Konular:** Clustering (Kümeleme) + Data Visualization (Görselleştirme)")

# --- DOSYA YÖNETİMİ ---
DATA_FILE = "src/social.csv"
MODEL_FILE = "src/kmeans_model.pkl"
SCALER_FILE = "src/cluster_scaler.pkl"
CLUSTERED_FILE = "src/social_clustered.csv"

def train_model():
    if not os.path.exists(DATA_FILE):
        return None, None, None
    
    df = pd.read_csv(DATA_FILE)
    X = df[['Age', 'EstimatedSalary']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    df['Cluster'] = kmeans.labels_
    
    joblib.dump(kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    df.to_csv(CLUSTERED_FILE, index=False)
    
    return kmeans, scaler, df

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(CLUSTERED_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE), pd.read_csv(CLUSTERED_FILE)
    else:
        return train_model()

kmeans, scaler, df = load_assets()

if df is None:
    st.error("Veri dosyası bulunamadı. Lütfen 'social.csv' yükleyin.")
    st.stop()

# --- ARAYÜZ VE GÖRSELLEŞTİRME ---

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Yeni Müşteri Ekle")
    age = st.number_input("Yaş", 18, 70, 30)
    salary = st.number_input("Maaş ($)", 15000, 150000, 50000, step=1000)
    
    # Kullanıcıdan alınan verinin kümesini tahmin et
    input_data = scaler.transform([[age, salary]])
    prediction = kmeans.predict(input_data)[0]
    
    if st.button("Hangi Kümeye Ait?"):
        st.success(f"Bu müşteri **Küme {prediction}** grubuna aittir.")

with col2:
    st.subheader("İnteraktif Segmentasyon Analizi")
    
    # Görselleştirme Seçeneği
    viz_type = st.radio("Grafik Türü:", ["2D Scatter Plot", "3D Scatter Plot"], horizontal=True)
    
    if viz_type == "2D Scatter Plot":
        fig = px.scatter(
            df, x='Age', y='EstimatedSalary', 
            color='Cluster', symbol='Cluster',
            title="Müşteri Segmentleri (Yaş vs Maaş)",
            color_continuous_scale=px.colors.qualitative.Bold
        )
        # 2D İşaretçi
        fig.add_trace(go.Scatter(
            x=[age], y=[salary], mode='markers',
            marker=dict(color='black', size=15, symbol='x'),
            name='Siz (Seçilen)'
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "3D Scatter Plot":
        # 3. Boyut: Purchased (0 veya 1)
        fig = px.scatter_3d(
            df, x='Age', y='EstimatedSalary', z='Purchased',
            color='Cluster',
            title="3D Analiz: Yaş - Maaş - Satın Alma Durumu",
            opacity=0.7
        )
        
        # 3D İşaretçi (DÜZELTME BURADA YAPILDI)
        # Z eksenine 0.5 veriyoruz ki tam ortada havada asılı dursun, dikkat çeksin.
        fig.add_trace(go.Scatter3d(
            x=[age], y=[salary], z=[0.5], 
            mode='markers',
            marker=dict(color='black', size=10, symbol='x'),
            name='Siz (Konumunuz)'
        ))
        
        st.plotly_chart(fig, use_container_width=True)

# --- İSTATİSTİKLER ---
st.divider()
st.subheader("Küme İstatistikleri")
cluster_stats = df.groupby('Cluster')[['Age', 'EstimatedSalary', 'Purchased']].mean()
st.dataframe(cluster_stats, use_container_width=True)
st.caption("Not: Purchased sütununda 1'e yakın değerler, o kümenin satın alma oranının yüksek olduğunu gösterir.")