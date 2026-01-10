import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------
# MODELİ YÜKLE
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("src/pet_segmentation_unet.keras")

model = load_model()

IMG_SIZE = 128

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return image

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🐶🐱 Pet Image Segmentation")
st.write("Semantic Segmentation using U-Net (Oxford-IIIT Pet Dataset)")

uploaded_file = st.file_uploader(
    "Upload an image (cat or dog)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_image = preprocess_image(image)

    # Prediction
    pred = model.predict(input_image[np.newaxis, ...])
    pred_mask = np.argmax(pred[0], axis=-1)

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pred_mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    # Overlay
    overlay = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    overlay[pred_mask == 2] = [0, 0, 0]  # background black

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    st.pyplot(fig)
