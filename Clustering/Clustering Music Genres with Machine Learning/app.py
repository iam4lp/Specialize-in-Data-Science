# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Music Recommendation System", layout="wide")
st.title("🎵 Spotify Music Recommendation & Clustering")

# --- DOSYALARI YÜKLE ---
@st.cache_data
def load_data():
    # İşlenmiş veriyi yükle (train_model.py çalışınca oluşur)
    possible_paths = ["spotify_clustered.csv", "src/spotify_clustered.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Eğer işlenmiş veri yoksa ham veriyi yükle (Fallback)
    if os.path.exists("Spotify-2000.csv"):
        st.warning("İşlenmiş veri bulunamadı, ham veri kullanılıyor. Lütfen önce 'train_model.py' dosyasını çalıştırın.")
        df = pd.read_csv("src/Spotify-2000.csv")
        return df
    
    return None

df = load_data()

if df is None:
    st.error("Veri dosyası (Spotify-2000.csv veya spotify_clustered.csv) bulunamadı.")
    st.stop()

# Küme bilgisi yoksa uyarı ver
if 'Cluster' not in df.columns:
    st.error("Veri setinde 'Cluster' sütunu yok. Lütfen modeli eğitin.")
    st.stop()

# --- ARAYÜZ ---

# 1. Kümeleme Görselleştirmesi
st.subheader("Müzik Kümeleri Haritası (PCA)")
st.markdown("Yapay zeka, şarkıları ses özelliklerine göre grupladı. Her nokta bir şarkıdır.")

fig = px.scatter(
    df, x="PCA1", y="PCA2", color="Cluster", 
    hover_data=["Title", "Artist", "Top Genre"],
    title="Şarkı Kümeleri Dağılımı"
)
st.plotly_chart(fig, use_container_width=True)

# 2. Şarkı Öneri Sistemi
st.subheader("🎧 Şarkı Öneri Motoru")
st.markdown("Sevdiğiniz bir şarkıyı seçin, size benzer şarkıları önerelim.")

# Şarkı Seçimi
song_list = df['Title'] + " - " + df['Artist']
selected_song_str = st.selectbox("Şarkı Seçiniz:", song_list)

if st.button("Öneri Yap"):
    # Seçilen şarkının bilgilerini bul
    selected_index = song_list[song_list == selected_song_str].index[0]
    selected_song = df.iloc[selected_index]
    
    selected_cluster = selected_song['Cluster']
    
    st.info(f"Seçilen Şarkı: **{selected_song['Title']}** ({selected_song['Artist']}) | Küme: {selected_cluster}")
    
    # Aynı kümedeki diğer şarkıları bul
    recommendations = df[df['Cluster'] == selected_cluster].sample(5) # Rastgele 5 tane getir
    
    st.write("---")
    st.write("### Sizin İçin Önerilenler:")
    
    cols = st.columns(5)
    for i, (_, row) in enumerate(recommendations.iterrows()):
        with cols[i]:
            st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=50) # Logo
            st.markdown(f"**{row['Title']}**")
            st.caption(f"{row['Artist']}")
            st.caption(f"Tür: {row['Top Genre']}")
            
    # Radar Grafiği (Seçilen şarkı vs Küme Ortalaması)
    st.write("---")
    st.write("### Neden Bu Şarkılar?")
    
    features = ['Energy', 'Danceability', 'Liveness', 'Valence', 'Acousticness']
    
    # Veriyi normalize et (0-1 arası) grafiğin düzgün görünmesi için
    # Basitçe 100'e bölelim (Çünkü veri setinde 0-100 arası genelde)
    # Veya min-max scaling yapılmış halini kullanabiliriz ama burada hızlıca görselleştirelim.
    
    cluster_mean = df[df['Cluster'] == selected_cluster][features].mean()
    song_values = selected_song[features]
    
    # Radar Chart Data
    radar_df = pd.DataFrame(dict(
        r=song_values.values,
        theta=features
    ))
    fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, title="Şarkının Ses Profili")
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar)