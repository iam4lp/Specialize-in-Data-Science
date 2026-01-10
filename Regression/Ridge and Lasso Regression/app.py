# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Ridge vs Lasso", layout="wide")
st.title("📉 Ridge & Lasso Regression")
st.markdown("Reklam bütçesi optimizasyonu ve **Regularization** etkisi.")

# --- OTOMATİK EĞİTİM MODÜLÜ ---
DATA_FILE = "src/Advertising.csv"
RIDGE_FILE = "src/ridge_model.pkl"
LASSO_FILE = "src/lasso_model.pkl"
SCALER_FILE = "src/ads_scaler.pkl"

def train_models():
    if not os.path.exists(DATA_FILE):
        return None, None, None
        
    df = pd.read_csv(DATA_FILE)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    
    joblib.dump(ridge, RIDGE_FILE)
    joblib.dump(lasso, LASSO_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    return ridge, lasso, scaler

@st.cache_resource
def load_assets():
    if os.path.exists(RIDGE_FILE) and os.path.exists(LASSO_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(RIDGE_FILE), joblib.load(LASSO_FILE), joblib.load(SCALER_FILE)
    else:
        return train_models()

ridge_model, lasso_model, scaler = load_assets()

if ridge_model is None:
    st.error("Veri dosyası bulunamadı. Lütfen 'Advertising.csv' yükleyin.")
    st.stop()

# --- ARAYÜZ ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Bütçe Planlama ($)")
    tv = st.number_input("TV Reklamı", 0, 500, 150)
    radio = st.number_input("Radyo Reklamı", 0, 100, 20)
    newspaper = st.number_input("Gazete Reklamı", 0, 100, 10)
    
    predict_btn = st.button("Satış Tahmini Yap")

# --- SONUÇLAR VE GRAFİK ---
if predict_btn:
    input_data = scaler.transform([[tv, radio, newspaper]])
    
    pred_ridge = ridge_model.predict(input_data)[0]
    pred_lasso = lasso_model.predict(input_data)[0]
    
    with col2:
        st.subheader("Tahmin Sonuçları")
        c1, c2 = st.columns(2)
        c1.metric("Ridge Tahmini (Satış)", f"{pred_ridge:.2f}k Birim")
        c2.metric("Lasso Tahmini (Satış)", f"{pred_lasso:.2f}k Birim")
        
        st.divider()
        st.subheader("Model Katsayıları (Feature Importance)")
        st.markdown("Lasso'nun gereksiz özellikleri nasıl **sıfırladığına** dikkat edin!")
        
        # Katsayıları Karşılaştırma Grafiği
        features = ['TV', 'Radio', 'Newspaper']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features, y=ridge_model.coef_,
            name='Ridge (L2)', marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=features, y=lasso_model.coef_,
            name='Lasso (L1)', marker_color='red'
        ))
        
        fig.update_layout(barmode='group', title="Hangi Reklam Daha Etkili?")
        st.plotly_chart(fig, use_container_width=True)
        
        if lasso_model.coef_[2] == 0:
            st.info("ℹ️ **Lasso Analizi:** 'Newspaper' reklamının etkisi 0'a indirilmiş. Yani bu mecra satışları etkilemiyor, bütçe ayırmanıza gerek yok!")