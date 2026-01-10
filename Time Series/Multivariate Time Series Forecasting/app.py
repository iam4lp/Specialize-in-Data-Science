import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

st.set_page_config(page_title="Stock Price Forecasting", layout="wide")
st.title("📈 Multivariate Time Series: Stock Prediction")

# --- PARAMETRELER ---
SEQ_LENGTH = 60
MODEL_PATH = "src/stock_lstm_model.h5"
SCALER_PATH = "src/stock_scaler.pkl"
DATA_PATH = "src/stocks.csv"

# --- YARDIMCI FONKSİYONLAR ---

def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 4]) # 4: Close
    return np.array(X), np.array(y)

def train_new_model():
    """Modeli sunucuda sıfırdan eğitir"""
    status_text = st.empty()
    status_text.warning("⚠️ Yüklenen model uyumsuz. Sunucuda yeni model eğitiliyor... (Bu işlem 30-60 sn sürebilir)")
    progress_bar = st.progress(0)
    
    # 1. Veri Hazırlığı
    df = pd.read_csv(DATA_PATH)
    data = df[df['Ticker'] == 'GOOG'].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    dataset = data[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # 2. Model Mimarisi
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 3. Eğitim
    progress_bar.progress(30)
    # Hızlı eğitim için epoch sayısını düşük tutuyoruz (Deployment için yeterli)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    progress_bar.progress(90)
    
    # 4. Kaydetme
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    progress_bar.progress(100)
    status_text.success("✅ Yeni model başarıyla eğitildi ve yüklendi!")
    return model, scaler

@st.cache_resource
def load_smart_assets():
    # 1. Veri Seti Kontrolü
    if not os.path.exists(DATA_PATH):
        return None, None, "DATA_MISSING"

    model = None
    scaler = None
    
    # 2. Modeli Yüklemeyi Dene
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Eski model yüklenemedi: {e}")
            model = None # Hata varsa None yap ki aşağıda yeniden eğitsin
    
    # 3. Scaler Yüklemeyi Dene
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except:
            scaler = None

    # 4. Eğer Model veya Scaler yoksa/bozuksa -> YENİDEN EĞİT
    if model is None or scaler is None:
        model, scaler = train_new_model()
        
    return model, scaler, "OK"

# --- UYGULAMA AKIŞI ---

model, scaler, status = load_smart_assets()

if status == "DATA_MISSING":
    st.error("❌ 'stocks.csv' dosyası bulunamadı! Lütfen 'Files' sekmesinden dosyayı yükleyin.")
    st.stop()

# --- VERİ YÜKLEME ---
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# --- ARAYÜZ VE TAHMİN ---
st.sidebar.divider()
st.sidebar.header("Ayarlar")
ticker_list = df['Ticker'].unique()
selected_ticker = st.sidebar.selectbox("Hisse Senedi Seç", ticker_list, index=list(ticker_list).index('GOOG') if 'GOOG' in ticker_list else 0)

data_ticker = df[df['Ticker'] == selected_ticker].sort_values('Date')

st.subheader(f"{selected_ticker} - Tarihsel Fiyat Grafiği")
fig = go.Figure(data=[go.Candlestick(x=data_ticker['Date'],
                open=data_ticker['Open'],
                high=data_ticker['High'],
                low=data_ticker['Low'],
                close=data_ticker['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Gelecek Fiyat Tahmini (AI Model)")

if selected_ticker != 'GOOG':
    st.caption("Not: Model GOOG verisi ile eğitilmiştir, diğer hisseler için tahminler yaklaşık değerlerdir.")

last_60_days = data_ticker[['Open', 'High', 'Low', 'Volume', 'Close']].tail(60).values

if len(last_60_days) == 60:
    try:
        last_60_scaled = scaler.transform(last_60_days)
        X_input = np.array([last_60_scaled])
        predicted_scaled = model.predict(X_input)
        
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 4] = predicted_scaled[0, 0]
        
        predicted_price = scaler.inverse_transform(dummy_array)[0, 4]
        last_close = data_ticker['Close'].iloc[-1]
        
        change = predicted_price - last_close
        change_pct = (change / last_close) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Son Kapanış", f"${last_close:.2f}")
        col2.metric("Tahmini Fiyat (Gelecek)", f"${predicted_price:.2f}", f"{change_pct:.2f}%")
        
        if change > 0:
            st.success("📈 Model Yükseliş Öngörüyor!")
        else:
            st.error("📉 Model Düşüş Öngörüyor!")
            
    except Exception as e:
        st.error(f"Tahmin yapılırken hata oluştu: {e}")
else:
    st.warning("Tahmin için yeterli veri yok.")