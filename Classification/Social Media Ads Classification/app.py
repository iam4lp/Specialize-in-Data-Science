# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Social Media Ads Predictor", layout="wide")
st.title("📱 Social Media Ads Classification")
st.markdown("Kullanıcının **Yaşına** ve **Maaşına** göre reklamdaki ürünü satın alıp almayacağını tahmin eder.")

# --- MODEL YÜKLEME (OTOMATİK EĞİTİM MODLU) ---
DATA_FILE = "src/social.csv"
MODEL_FILE = "src/social_model.pkl"
SCALER_FILE = "src/social_scaler.pkl"

def train_model():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    if not os.path.exists(DATA_FILE):
        return None, None
        
    df = pd.read_csv(DATA_FILE)
    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_scaled, y)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    else:
        return train_model()

model, scaler = load_assets()

if model is None:
    st.error("Veri dosyası bulunamadı. Lütfen 'social.csv' dosyasını yükleyin.")
    st.stop()

# --- ARAYÜZ ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Kullanıcı Profili")
    age = st.slider("Yaş", 18, 60, 30)
    salary = st.slider("Tahmini Maaş ($)", 15000, 150000, 50000, step=500)
    
    predict_btn = st.button("Satın Alma Tahmini Yap")

with col2:
    # --- KARAR SINIRI GÖRSELLEŞTİRME ---
    st.header("Model Karar Sınırları")
    
    # Veriyi yükle (Görselleştirme için)
    df = pd.read_csv(DATA_FILE)
    
    # Meshgrid oluştur (Arka planı boyamak için)
    x_min, x_max = df['Age'].min() - 5, df['Age'].max() + 5
    y_min, y_max = df['EstimatedSalary'].min() - 5000, df['EstimatedSalary'].max() + 5000
    
    # Kullanıcının girdiği nokta
    user_input = pd.DataFrame({'Age': [age], 'EstimatedSalary': [salary]})
    
    # Scatter Plot
    fig = px.scatter(df, x='Age', y='EstimatedSalary', color=df['Purchased'].astype(str),
                     color_discrete_map={'0': 'red', '1': 'green'},
                     labels={'0': 'Almadı', '1': 'Aldı'},
                     title="Müşteri Dağılımı ve Sizin Konumunuz")
    
    # Kullanıcının yerini işaretle
    fig.add_trace(go.Scatter(x=[age], y=[salary], mode='markers', 
                             marker=dict(color='blue', size=15, symbol='x'),
                             name='Siz'))
    
    st.plotly_chart(fig, use_container_width=True)

# --- TAHMİN SONUCU ---
if predict_btn:
    # Ölçeklendirme
    input_scaled = scaler.transform([[age, salary]])
    
    # Tahmin
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] # Satın alma olasılığı
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.success(f"✅ **SONUÇ: SATIN ALIR**")
            st.metric("Olasılık", f"%{probability*100:.1f}")
        else:
            st.error(f"❌ **SONUÇ: SATIN ALMAZ**")
            st.metric("Satın Alma Olasılığı", f"%{probability*100:.1f}")
            
    with res_col2:
        if prediction == 1:
            st.info("Bu kullanıcı profili, hedef kitleye uygundur. Reklam gösterilebilir.")
        else:
            st.warning("Bu kullanıcı profili ilgisiz görünüyor. Reklam bütçesi harcanmamalı.")