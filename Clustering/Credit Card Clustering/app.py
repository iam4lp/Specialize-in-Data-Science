# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Credit Card Clustering", layout="wide")
st.title("💳 Credit Card Customer Segmentation")

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    if os.path.exists("src/credit_card_clustered.csv"):
        return pd.read_csv("src/credit_card_clustered.csv")
    return None

df = load_data()

# Skor Yükleme
score = "Hesaplanmadı"
if os.path.exists("src/model_score.txt"):
    with open("src/model_score.txt", "r") as f:
        score = f.read()

if df is None:
    st.error("Veri dosyası bulunamadı. Lütfen önce 'train_model.py' dosyasını çalıştırın.")
    st.stop()

# --- METRİKLER ---
st.sidebar.header("Model Performansı")
st.sidebar.metric("Silhouette Skoru", score)
st.sidebar.info("Silhouette Score: Kümeleme kalitesini gösterir. (1: Mükemmel, 0: Kötü, -1: Yanlış)")

n_clusters = df['Cluster'].nunique()
st.sidebar.write(f"**Toplam Küme Sayısı:** {n_clusters}")

# --- GÖRSELLEŞTİRME ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Müşteri Segmentleri Haritası (PCA)")
    fig = px.scatter(
        df, x="PCA1", y="PCA2", color="Cluster",
        title="Müşteri Kümeleri Dağılımı",
        opacity=0.7,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Küme Dağılımı")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_pie = px.pie(cluster_counts, values='Count', names='Cluster', title="Müşteri Oranları")
    st.plotly_chart(fig_pie, use_container_width=True)

# --- KÜME ANALİZİ (PROFILING) ---
st.divider()
st.subheader("🔍 Küme Karakteristikleri (Ortalama Değerler)")

# Sayısal olmayan sütunları çıkar (Varsa)
numeric_cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
# Cluster'a göre grupla ve ortalamasını al
cluster_profile = df.groupby('Cluster')[numeric_cols].mean().reset_index()

st.dataframe(cluster_profile.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

# Yorumlama Kılavuzu (Otomatik Analiz)
st.write("### 💡 Segment Yorumları")
for i, row in cluster_profile.iterrows():
    cluster_id = int(row['Cluster'])
    balance = row['BALANCE']
    purchases = row['PURCHASES']
    cash_adv = row['CASH_ADVANCE']
    
    label = "Standart Müşteri"
    if purchases > 2000:
        label = "💰 Büyük Harcamacılar (Big Spenders)"
    elif cash_adv > 2000:
        label = "💸 Nakit Avansçılar (Cash Advance Users)"
    elif balance > 3000 and purchases < 500:
        label = "⚠️ Yüksek Bakiyeli / Az Harcayanlar (Riskli?)"
    elif purchases < 500 and cash_adv < 500:
        label = "📉 Düşük Aktivite / Tutumlular"
        
    st.info(f"**Küme {cluster_id}:** {label} (Ort. Bakiye: ${balance:.0f}, Ort. Harcama: ${purchases:.0f})")