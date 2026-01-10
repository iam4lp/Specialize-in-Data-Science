import streamlit as st
import pickle
import numpy as np

# -------------------------------------------------
# MODELİ YÜKLE
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("src/amazon_item_based_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()

item_user_matrix = data["item_user_matrix"]
item_similarity = data["item_similarity"]
user_categories = data["user_categories"]
product_categories = data["product_categories"]


# -------------------------------------------------
# STREAMLIT ARAYÜZ
# -------------------------------------------------
st.title("🛒 Amazon Recommendation System")
st.write("Item-Based Collaborative Filtering")

user_id = st.selectbox("Select a User ID", user_categories)
top_n = st.slider("Number of recommendations", 3, 10, 5)

# -------------------------------------------------
# RECOMMENDATION LOGIC (HF SAFE)
# -------------------------------------------------
if st.button("Recommend Products"):

    user_idx = user_categories.get_loc(user_id)
    user_ratings = item_user_matrix[:, user_idx].toarray().flatten()

    rated_items_idx = np.where(user_ratings > 0)[0]

    if len(rated_items_idx) == 0:
        st.warning("This user has no ratings.")
    else:
        scores = {}

        for item_idx in rated_items_idx:
            similarities = item_similarity[item_idx].toarray().flatten()

            for sim_idx, sim_score in enumerate(similarities):
                if sim_idx not in rated_items_idx and sim_score > 0:
                    scores[sim_idx] = scores.get(sim_idx, 0) + sim_score

        if not scores:
            st.warning("No recommendations could be generated.")
        else:
            ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

            st.subheader("🔎 Recommended Products")
            for idx, score in ranked_items:
                product_id = product_categories[idx]
                st.write(f"**Product ID:** {product_id} — Similarity Score: {score:.3f}")