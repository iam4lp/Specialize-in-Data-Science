import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

# Sayfa Ayarları
st.set_page_config(page_title="Data Augmentation Lab Pro", layout="wide")
st.title("🧬 Data Augmentation Lab: Advanced Features")
st.markdown("""
Bu proje, **TensorFlow Keras Preprocessing Layers** kullanarak veri çoğaltma tekniklerini analiz eder.
Artık **Grayscale**, **Saturation** ve **Brightness** özellikleri de eklendi!
""")

# --- BÖLÜM 1: AUGMENTATION MODELİ OLUŞTURMA ---
st.sidebar.header("Augmentation Parametreleri")

# Geometrik Dönüşümler
st.sidebar.subheader("📍 Geometrik")
flip_mode = st.sidebar.selectbox("Random Flip", ["horizontal", "vertical", "horizontal_and_vertical"], index=0)
rotation_factor = st.sidebar.slider("Random Rotation", 0.0, 1.0, 0.2)
zoom_factor = st.sidebar.slider("Random Zoom", 0.0, 1.0, 0.2)

# Renk ve Işık Dönüşümleri (YENİ EKLENENLER)
st.sidebar.subheader("🎨 Renk ve Işık")
contrast_factor = st.sidebar.slider("Random Contrast", 0.0, 1.0, 0.2)
brightness_factor = st.sidebar.slider("Random Brightness", 0.0, 1.0, 0.2) # Yeni
saturation_factor = st.sidebar.slider("Random Saturation", 0.0, 5.0, 1.0) # Yeni (1.0 = orijinal civarı)
grayscale_prob = st.sidebar.checkbox("Apply Grayscale (Gri Tonlama)", value=False) # Yeni

# Özel Lambda Katmanları
def random_saturation_layer(x, factor):
    # Factor 0 ise değişiklik yok, yüksekse çok doygun/az doygun rastgele seçer
    return tf.image.random_saturation(x, lower=max(0.1, 1.0-factor), upper=1.0+factor)

def to_grayscale_layer(x):
    # Grayscale'e çevirip tekrar 3 kanala yapıyoruz ki diğer katmanlarla uyumlu olsun
    gray = tf.image.rgb_to_grayscale(x)
    return tf.image.grayscale_to_rgb(gray)

def get_augmentation_model(flip, rot, zoom, contrast, brightness, saturation, gray_on):
    layer_list = [
        layers.RandomFlip(flip),
        layers.RandomRotation(rot),
        layers.RandomZoom(zoom),
        layers.RandomContrast(contrast),
        layers.RandomBrightness(brightness) # Parlaklık Katmanı
    ]
    
    # Saturation için Lambda Katmanı (Eğer faktör > 0 ise ekle)
    if saturation > 0:
        layer_list.append(layers.Lambda(lambda x: random_saturation_layer(x, saturation)))
        
    # Grayscale (Eğer seçiliyse ekle)
    if gray_on:
        layer_list.append(layers.Lambda(lambda x: to_grayscale_layer(x)))

    data_augmentation = Sequential(layer_list, name="data_augmentation")
    return data_augmentation

# Modeli oluştur
aug_model = get_augmentation_model(
    flip_mode, rotation_factor, zoom_factor, 
    contrast_factor, brightness_factor, saturation_factor, grayscale_prob
)

# --- BÖLÜM 2: GÖRSEL TEST LABORATUVARI ---
col1, col2 = st.columns(2)

uploaded_file = st.file_uploader("Bir resim yükleyin (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Resmi Yükle ve İşle
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image)
    img_tensor = tf.expand_dims(img_array, 0) # (1, 256, 256, 3)

    with col1:
        st.subheader("Orijinal")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Augmented (İşlenmiş)")
        if st.button("Varyasyon Üret 🎲", key="btn1"):
            # Augmentation'ı uygula
            # Not: Lambda katmanları bazen cast gerektirebilir, float32'ye çevirelim
            img_tensor_float = tf.cast(img_tensor, tf.float32)
            
            # Augmented görüntü (0-255 arası değerler float olabilir)
            augmented_image = aug_model(img_tensor_float, training=True)
            
            # Görüntüleme için uint8'e geri çevir ve clip yap (taşmaları önle)
            result = tf.clip_by_value(augmented_image[0], 0, 255)
            result_uint8 = tf.cast(result, tf.uint8).numpy()
            
            st.image(result_uint8, use_container_width=True)
            st.info("Her tıklamada parametre aralığında rastgele bir görüntü üretilir.")

# --- BİLGİ ALANI ---
with st.expander("📚 Parametreler Ne İşe Yarar?"):
    st.markdown("""
    * **Random Brightness:** Işık koşullarını simüle eder (Gündüz/Gece çekimleri için).
    * **Random Saturation:** Renklerin soluk veya çok canlı olmasını sağlar.
    * **Grayscale:** Renk bilgisini tamamen atar (Modelin renge değil şekle odaklanmasını sağlar).
    * **Random Contrast:** Gölge ve parlak alanlar arasındaki farkı değiştirir.
    """)