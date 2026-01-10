import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import re
import os
import matplotlib.pyplot as plt

# Hata ayıklama: WordCloud yüklü mü kontrol et
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPool1D

st.set_page_config(page_title="Ultimate NLP Analysis", layout="wide")
st.title("🧠 Ultimate NLP: Classic vs Deep Learning")

# --- SABİTLER ---
DATA_FILE = "IMDB Dataset.csv"
CLASSIC_MODEL_FILE = "nlp_classic_model_v2.pkl"
TFIDF_FILE = "nlp_tfidf_v2.pkl"
DL_MODEL_FILE = "nlp_dl_model_v2.h5"
TOKENIZER_FILE = "nlp_tokenizer_v2.pkl"

VOCAB_SIZE = 5000
MAX_LEN = 150
EMBEDDING_DIM = 128
SAMPLE_SIZE = 8000

# --- YARDIMCI FONKSİYONLAR ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(f"src/{filename}"):
        return f"src/{filename}"
    return None

def train_models_automatically():
    status_placeholder = st.empty()
    status_placeholder.warning("⚠️ Model güncelleniyor (V2)... Lütfen bekleyin (1-2 dk).")
    progress_bar = st.progress(0)
    
    csv_path = get_file_path(DATA_FILE)
    if not csv_path:
        st.error(f"❌ Veri seti ({DATA_FILE}) bulunamadı! Lütfen dosyayı yükleyin.")
        st.stop()
        
    df = pd.read_csv(csv_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['review'] = df['review'].apply(clean_text)
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
        
    X = df['review']
    y = df['sentiment']
    
    progress_bar.progress(10)
    
    # Model 1
    tfidf = TfidfVectorizer(max_features=VOCAB_SIZE)
    X_tfidf = tfidf.fit_transform(X)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_tfidf, y)
    
    joblib.dump(lr_model, CLASSIC_MODEL_FILE)
    joblib.dump(tfidf, TFIDF_FILE)
    progress_bar.progress(40)
    
    # Model 2
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_pad, y, epochs=7, batch_size=32, verbose=0)
    
    model.save(DL_MODEL_FILE)
    joblib.dump(tokenizer, TOKENIZER_FILE)
    
    progress_bar.progress(100)
    status_placeholder.success("✅ Modeller Hazır!")
    
    return lr_model, tfidf, model, tokenizer

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_assets():
    path_classic = get_file_path(CLASSIC_MODEL_FILE)
    path_tfidf = get_file_path(TFIDF_FILE)
    path_dl = get_file_path(DL_MODEL_FILE)
    path_tok = get_file_path(TOKENIZER_FILE)
    
    if path_classic and path_tfidf and path_dl and path_tok:
        try:
            classic_model = joblib.load(path_classic)
            tfidf = joblib.load(path_tfidf)
            dl_model = tf.keras.models.load_model(path_dl, compile=False)
            tokenizer = joblib.load(path_tok)
            return classic_model, tfidf, dl_model, tokenizer
        except:
            pass
            
    return train_models_automatically()

classic_model, tfidf, dl_model, tokenizer = load_assets()

# --- ARAYÜZ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Yorum Analizi")
    user_input = st.text_area("Analiz edilecek yorumu girin (İngilizce):", height=150, 
                              value="The movie was absolutely fantastic! The acting was great and the plot was moving.")
    
    analyze_btn = st.button("Analiz Et 🚀", type="primary")

with col2:
    st.info("💡 **Bilgi:**")
    st.write(f"Eğitim Verisi: **{SAMPLE_SIZE}** yorum")
    st.write("Model Versiyonu: **V2**")

if analyze_btn and user_input:
    cleaned_input = clean_text(user_input)
    
    # Tahminler
    vect_input = tfidf.transform([cleaned_input])
    pred_classic_prob = classic_model.predict_proba(vect_input)[0][1]
    
    seq_input = tokenizer.texts_to_sequences([cleaned_input])
    pad_input = pad_sequences(seq_input, maxlen=MAX_LEN, padding='post', truncating='post')
    pred_dl_prob = dl_model.predict(pad_input)[0][0]
    
    # Sonuçlar
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🏛️ Klasik Model")
        if pred_classic_prob > 0.5:
            st.success(f"Pozitif (%{pred_classic_prob*100:.1f})")
        else:
            st.error(f"Negatif (%{(1-pred_classic_prob)*100:.1f})")
        st.progress(float(pred_classic_prob))
        
    with c2:
        st.subheader("🧠 Deep Learning")
        if 0.45 < pred_dl_prob < 0.55:
             st.warning(f"Kararsız (%{pred_dl_prob*100:.1f})")
        elif pred_dl_prob > 0.5:
            st.success(f"Pozitif (%{pred_dl_prob*100:.1f})")
        else:
            st.error(f"Negatif (%{(1-pred_dl_prob)*100:.1f})")
        st.progress(float(pred_dl_prob))

    # --- WORD CLOUD (Hata Ayıklama Modu) ---
    st.divider()
    st.subheader("☁️ Kelime Bulutu")
    
    if not WORDCLOUD_AVAILABLE:
        st.error("❌ 'wordcloud' kütüphanesi yüklü değil! Lütfen requirements.txt dosyasını kontrol edin.")
    else:
        try:
            # WordCloud oluşturma
            wordcloud = WordCloud(width=800, height=300, background_color='white').generate(user_input)
            
            # Çizim
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
        except Exception as e:
            # GERÇEK HATAYI GÖSTER
            st.error(f"❌ WordCloud Oluşturma Hatası: {e}")
            st.info("Lütfen requirements.txt dosyanızda 'wordcloud' ve 'matplotlib' olduğundan emin olun.")