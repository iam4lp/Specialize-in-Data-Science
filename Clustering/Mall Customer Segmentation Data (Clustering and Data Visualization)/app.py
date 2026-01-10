import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="centered"
)

st.title("🛍️ Customer Segmentation (Clustering + Visualization)")
st.write(
    """
This application uses a **pre-trained K-Means clustering model**
to segment customers based on their income and spending behavior.
"""
)

# --------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("src/Mall_Customers.csv")

@st.cache_resource
def load_model():
    kmeans = joblib.load("src/kmeans_model.pkl")
    scaler = joblib.load("src/scaler.pkl")
    return kmeans, scaler

df = load_data()
kmeans, scaler = load_model()

# --------------------------------------------------
# FEATURE PREPARATION
# --------------------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
X_scaled = scaler.transform(X)

# --------------------------------------------------
# PREDICT CLUSTERS
# --------------------------------------------------
df["Cluster"] = kmeans.predict(X_scaled)

# --------------------------------------------------
# SILHOUETTE SCORE (DISPLAY ONLY)
# --------------------------------------------------
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_scaled, df["Cluster"])

st.subheader("📊 Model Performance")
st.write(f"**Silhouette Score:** `{sil_score:.3f}`")

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
st.subheader("📈 Customer Segmentation Visualization")

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    data=df,
    ax=ax
)

ax.set_title("Customer Segments using K-Means")
st.pyplot(fig)

# --------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------
st.subheader("📌 Cluster Summary")

summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

st.dataframe(summary)

# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------
st.subheader("🧠 Interpretation")

st.write(
    """
- Customers are grouped using **unsupervised learning (K-Means)**
- Clusters represent different **spending behaviors**
- Results can support **marketing and business decision-making**
"""
)