import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import re
import os
import difflib # YENİ: Yaklaşık eşleşme bulmak için

st.set_page_config(page_title="Netflix Recommender", layout="wide")
st.title("🎬 Netflix Recommendation System")

# --- SABİTLER ---
DATA_FILE = "src/netflixData.csv"
PROCESSED_FILE = "src/netflix_processed.csv"
SIM_FILE = "src/netflix_cosine_sim.pkl"
INDICES_FILE = "src/netflix_indices.pkl"

# --- EĞİTİM VE HAZIRLIK FONKSİYONLARI ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def prepare_data():
    """Veriyi işler ve benzerlik matrisini oluşturur (Sunucuda çalışır)"""
    status = st.empty()
    status.warning("⚠️ Model dosyaları eksik. Sistem kendini hazırlıyor... (Bu işlem 1-2 dakika sürebilir)")
    progress = st.progress(0)
    
    # 1. Veri Yükleme
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ '{DATA_FILE}' bulunamadı! Lütfen Files sekmesinden CSV dosyasını yükleyin.")
        st.stop()
        
    df = pd.read_csv(DATA_FILE)
    progress.progress(20)
    
    # 2. Temizleme
    df['Description'] = df['Description'].fillna('')
    df['Genres'] = df['Genres'].fillna('')
    df['Director'] = df['Director'].fillna('')
    df['Cast'] = df['Cast'].fillna('')
    
    # Özellikleri Birleştirme
    df['combined_features'] = (
        df['Description'] + " " + 
        df['Genres'] + " " + 
        df['Director'] + " " + 
        df['Cast']
    )
    df['combined_features'] = df['combined_features'].apply(clean_text)
    progress.progress(50)
    
    # 3. TF-IDF ve Benzerlik Matrisi
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    progress.progress(80)
    
    # 4. Kaydetme
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
    
    joblib.dump(cosine_sim, SIM_FILE)
    joblib.dump(indices, INDICES_FILE)
    df.to_csv(PROCESSED_FILE, index=False)
    
    progress.progress(100)
    status.success("✅ Sistem başarıyla hazırlandı!")
    return df, cosine_sim, indices

# --- VERİ YÜKLEME ---
@st.cache_resource
def load_data_smart():
    if os.path.exists(PROCESSED_FILE) and os.path.exists(SIM_FILE) and os.path.exists(INDICES_FILE):
        try:
            df = pd.read_csv(PROCESSED_FILE)
            cosine_sim = joblib.load(SIM_FILE)
            indices = joblib.load(INDICES_FILE)
            return df, cosine_sim, indices
        except:
            pass 
    return prepare_data()

df, cosine_sim, indices = load_data_smart()

# --- ÖNERİ FONKSİYONU ---
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = indices[title]
    except KeyError:
        return []

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# --- ARAYÜZ ---
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", use_container_width=True)

with col2:
    st.write("")
    st.markdown("### İçerik Bazlı Öneri Motoru")
    st.markdown("İster listeden seçin, ister adını yazın; yapay zeka size en benzerlerini bulsun.")

st.divider()

# --- SEÇİM ALANI (YENİLENMİŞ) ---
movie_list = df['Title'].unique()

# Arama Yöntemi Seçimi
search_method = st.radio("Arama Yöntemi:", ("🔽 Listeden Seç", "✍️ İsmini Yazarak Ara"), horizontal=True)

selected_movie = None

if search_method == "🔽 Listeden Seç":
    selected_movie = st.selectbox("Bir içerik seçin:", movie_list)

else:
    # Yazarak Arama Kısmı
    user_input = st.text_input("Film/Dizi adını yazın (Örn: Godfather, Squid...)", placeholder="Buraya yazın...")
    
    if user_input:
        # En yakın eşleşmeyi bul (Fuzzy Match)
        matches = difflib.get_close_matches(user_input, movie_list, n=1, cutoff=0.4)
        
        if matches:
            found_movie = matches[0]
            st.info(f"🔍 Bulunan en yakın sonuç: **{found_movie}**")
            selected_movie = found_movie
        else:
            st.warning("⚠️ Eşleşen bir içerik bulunamadı. Lütfen listeden kontrol edin.")

# --- SONUÇLAR ---
if selected_movie:
    if st.button(f"'{selected_movie}' İçin Önerileri Getir"):
        with st.spinner('Yapay zeka analiz yapıyor...'):
            recommendations = get_recommendations(selected_movie)
            
            if len(recommendations) == 0:
                st.warning("Bu içerik için yeterli veri bulunamadı.")
            else:
                st.success(f"**{selected_movie}** içeriğini sevenler bunları da sevdi:")
                st.write("")
                
                for i, row in recommendations.iterrows():
                    with st.expander(f"📌 {row['Title']}"):
                        c1, c2 = st.columns([1, 5])
                        with c2:
                            st.caption(f"**Tür:** {row['Genres']}  |  **Yıl:** {row['Release Date']}  |  **IMDb:** {row['Imdb Score']}")
                            st.write(f"_{row['Description']}_")
                            st.markdown(f"**Oyuncular:** {row['Cast']}")