import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

st.set_page_config(page_title="Airline Passengers Forecast", layout="wide")
st.title("✈️ Airline Passengers Forecasting with LSTM")

# --- SABİTLER ---
DATA_FILE = "src/airline-passengers.csv"
MODEL_FILE = "src/lstm_airline_model.h5"
SCALER_FILE = "src/airline_scaler.pkl"
LOOK_BACK = 3

# --- YARDIMCI FONKSİYONLAR ---
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_new_model(df):
    """Sunucuda sıfırdan model eğitir"""
    status = st.empty()
    status.warning("⚠️ Mevcut model uyumsuz. Sunucuda yeni model eğitiliyor... (Lütfen bekleyin)")
    progress = st.progress(0)
    
    # Veri Hazırlığı
    data = df.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    train_size = int(len(scaled_data) * 0.67)
    test_size = len(scaled_data) - train_size
    train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]
    
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    
    # Model Mimarisi
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, LOOK_BACK)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Eğitim (Hızlı olması için epoch sayısı optimize edildi)
    progress.progress(20)
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)
    progress.progress(90)
    
    # Kaydet
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    progress.progress(100)
    status.success("✅ Yeni model başarıyla eğitildi ve devreye alındı!")
    return model, scaler

# --- VERİ VE MODEL YÜKLEME ---
@st.cache_data
def load_data():
    # Dosya yollarını kontrol et (src içinde veya ana dizinde)
    if os.path.exists(DATA_FILE):
        path = DATA_FILE
    elif os.path.exists("src/" + DATA_FILE):
        path = "src/" + DATA_FILE
    else:
        return None
    
    df = pd.read_csv(path)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df

df = load_data()

if df is None:
    st.error(f"❌ '{DATA_FILE}' dosyası bulunamadı! Lütfen Files sekmesinden CSV dosyasını yükleyin.")
    st.stop()

@st.cache_resource
def load_smart_assets():
    # Önce var olanı yüklemeyi dene
    model = None
    scaler = None
    
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = tf.keras.models.load_model(MODEL_FILE, compile=False)
            scaler = joblib.load(SCALER_FILE)
        except Exception as e:
            # Hata verirse (sürüm uyuşmazlığı vb.) None döner, aşağıda yeniden eğitiriz
            pass
    
    # Model yüklenemediyse veya dosyalar yoksa YENİDEN EĞİT
    if model is None or scaler is None:
        model, scaler = train_new_model(df)
        
    return model, scaler

model, scaler = load_smart_assets()

# --- TAHMİN FONKSİYONU ---
def predict_future(model, scaler, data, look_back=3):
    dataset = data.values.astype('float32').reshape(-1, 1)
    dataset_scaled = scaler.transform(dataset)
    
    dataX = []
    for i in range(len(dataset_scaled)-look_back-1):
        a = dataset_scaled[i:(i+look_back), 0]
        dataX.append(a)
    
    if len(dataX) > 0:
        input_data = np.array(dataX)
        input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
        predictions_scaled = model.predict(input_data)
        predictions = scaler.inverse_transform(predictions_scaled)
        return predictions
    return []

# --- ARAYÜZ ---
st.subheader("Veri Genel Bakış")
col1, col2 = st.columns([3, 1])

with col1:
    st.line_chart(df['Passengers'])

with col2:
    st.write(df.tail())

st.divider()
st.subheader("LSTM Model Tahminleri")

if st.button("Tahminleri Çalıştır"):
    # Tahmin yap
    predictions = predict_future(model, scaler, df['Passengers'], LOOK_BACK)
    
    if len(predictions) > 0:
        # Grafiği hizalamak için
        # DÜZELTME BURADA YAPILDI: dtype=float eklendi
        plot_data = np.empty_like(df['Passengers'].values, dtype=float)
        plot_data[:] = np.nan
        
        start_idx = LOOK_BACK + 1
        end_idx = start_idx + len(predictions)
        
        # Grafik Çizimi
        pred_series = pd.Series(predictions.flatten(), index=df.index[start_idx:end_idx])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Passengers'], label='Gerçek Veri')
        ax.plot(pred_series.index, pred_series.values, label='LSTM Tahmini', color='red')
        ax.set_title("Gerçek vs Tahmin")
        ax.legend()
        st.pyplot(fig)
        
        # Gelecek Ay Tahmini
        last_window = df['Passengers'].values[-LOOK_BACK:].reshape(-1, 1)
        last_window_scaled = scaler.transform(last_window.astype('float32'))
        next_input = np.array([last_window_scaled.flatten()])
        next_input = np.reshape(next_input, (1, 1, LOOK_BACK))
        
        next_pred_scaled = model.predict(next_input)
        next_pred = scaler.inverse_transform(next_pred_scaled)
        
        st.success(f"📈 Gelecek Ay İçin Tahmini Yolcu Sayısı: **{int(next_pred[0][0])}**")
    else:
        st.error("Tahmin oluşturulamadı.")