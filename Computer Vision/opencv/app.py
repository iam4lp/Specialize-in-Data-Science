# app.py
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

st.set_page_config(page_title="Computer Vision: Face Privacy", layout="wide")
st.title("🕵️ Computer Vision: Face Anonymizer")
st.markdown("Bu uygulama, yüklenen fotoğraflardaki **yüzleri tespit eder** ve otomatik olarak **sansürler (blur)**.")

# --- SIDEBAR AYARLARI ---
st.sidebar.header("Ayarlar")
blur_rate = st.sidebar.slider("Bulanıklık Şiddeti", min_value=15, max_value=99, value=35, step=2)
# Kernel size tek sayı olmalıdır (OpenCV kuralı)
if blur_rate % 2 == 0:
    blur_rate += 1

detect_scale = st.sidebar.slider("Hassasiyet (Scale Factor)", 1.01, 1.5, 1.1)
min_neighbors = st.sidebar.slider("Komşuluk Sayısı (Min Neighbors)", 1, 10, 4)

# --- FONKSİYONLAR ---

@st.cache_resource
def load_face_cascade():
    # OpenCV'nin hazır Haar Cascade modelini yüklüyoruz
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

face_cascade = load_face_cascade()

def detect_and_blur(image_array, blur_k, scale_f, min_n):
    # Görüntüyü gri tona çevir (Tespiti hızlandırır)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_f, minNeighbors=min_n)
    
    # Orjinal görüntünün kopyasını al
    img_out = image_array.copy()
    
    for (x, y, w, h) in faces:
        # İlgi Alanını (ROI - Region of Interest) belirle
        roi = img_out[y:y+h, x:x+w]
        
        # Gaussian Blur uygula
        roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
        
        # Bulanık alanı ana görüntüye geri yapıştır
        img_out[y:y+h, x:x+w] = roi
        
        # Opsiyonel: Yüzün etrafına çerçeve çizmek istersen:
        # cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_out, len(faces)

# --- ANA AKIŞ ---

uploaded_file = st.file_uploader("Bir fotoğraf yükleyin (İnsan yüzü içeren)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Dosyayı oku
    image = Image.open(uploaded_file)
    image_array = np.array(image.convert('RGB')) # PIL -> Numpy
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orijinal Görüntü")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("İşlenmiş Görüntü")
        if st.button("Yüzleri Gizle"):
            processed_img, face_count = detect_and_blur(image_array, blur_rate, detect_scale, min_neighbors)
            
            st.image(processed_img, use_container_width=True)
            
            if face_count > 0:
                st.success(f"✅ Toplam **{face_count}** yüz tespit edildi ve sansürlendi.")
            else:
                st.warning("⚠️ Hiçbir yüz tespit edilemedi. Ayarları (Hassasiyet) değiştirmeyi deneyin.")
                
            # İndirme Butonu
            # Numpy array -> PIL Image -> Bytes
            result_image = Image.fromarray(processed_img)
            # İndirme işlemi için buffer vs gerekebilir ama Streamlit bunu kolaylaştırdı mı bakalım...
            # Basitçe kullanıcı sağ tıkla da kaydedebilir.