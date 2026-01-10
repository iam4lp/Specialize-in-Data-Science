import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="OpenCV Tutorial", layout="wide")
st.title("🎓 OpenCV with Python: Interactive Tutorial")
st.markdown("Bu proje, **Görüntü İşleme (Computer Vision)** tekniklerini interaktif olarak öğrenmeniz için tasarlanmıştır.")

# --- KENAR ÇUBUĞU (MENÜ) ---
st.sidebar.title("Ders Seçimi")
app_mode = st.sidebar.selectbox("Bir Konu Seçin:",
    ["1. Temel Filtreler & Efektler", 
     "2. Kenar Tespiti (Edge Detection)", 
     "3. Morfolojik İşlemler (Noise Removal)",
     "4. Yüz ve Göz Tespiti (Object Detection)"]
)

# --- RESİM YÜKLEME ---
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Bir Resim Yükleyin", type=["jpg", "jpeg", "png"])

# Varsayılan resim yoksa uyarı ver
if uploaded_file is None:
    st.info("Lütfen başlamak için sol menüden bir resim yükleyin. (İnsan yüzü içeren bir resim önerilir)")
    st.stop()

# Resmi Oku (PIL -> OpenCV Formatına Çevir)
original_image = np.array(Image.open(uploaded_file))
# OpenCV BGR kullanır, ama Streamlit RGB sever. İşlemleri RGB üzerinden yapacağız.
# Sadece cv2 fonksiyonlarına sokarken gerekirse griye çevireceğiz.

col1, col2 = st.columns(2)
with col1:
    st.subheader("Orijinal Görüntü")
    st.image(original_image, use_container_width=True)

# --- DERS 1: TEMEL FİLTRELER ---
if app_mode == "1. Temel Filtreler & Efektler":
    st.sidebar.subheader("Filtre Ayarları")
    filter_type = st.sidebar.radio("Efekt Seçin:", ["Grayscale (Gri)", "Sepia (Eskitme)", "Pencil Sketch (Karakalem)", "Blur (Bulanık)"])
    
    processed_image = original_image.copy()
    
    if filter_type == "Grayscale (Gri)":
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
    elif filter_type == "Sepia (Eskitme)":
        # Sepia Matrisi
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        processed_image = cv2.transform(original_image, kernel)
        processed_image = np.clip(processed_image, 0, 255) # Taşmaları önle
        
    elif filter_type == "Pencil Sketch (Karakalem)":
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        processed_image = cv2.divide(gray, 255 - blur, scale=256)
        
    elif filter_type == "Blur (Bulanık)":
        k_size = st.sidebar.slider("Bulanıklık Miktarı (Kernel Size)", 3, 51, 15, step=2)
        processed_image = cv2.GaussianBlur(original_image, (k_size, k_size), 0)

    with col2:
        st.subheader("İşlenmiş Görüntü")
        st.image(processed_image, use_container_width=True, channels="RGB" if len(processed_image.shape) == 3 else "GRAY")
        st.info(f"Uygulanan Efekt: **{filter_type}**")

# --- DERS 2: KENAR TESPİTİ ---
elif app_mode == "2. Kenar Tespiti (Edge Detection)":
    st.sidebar.subheader("Canny Ayarları")
    t_lower = st.sidebar.slider("Alt Eşik (Lower Threshold)", 0, 255, 50)
    t_upper = st.sidebar.slider("Üst Eşik (Upper Threshold)", 0, 255, 150)
    
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, t_lower, t_upper)
    
    with col2:
        st.subheader("Kenar Haritası")
        st.image(edges, use_container_width=True)
        st.caption("Canny algoritması, pikseller arasındaki ani renk değişimlerini (gradyanları) bularak kenarları çizer.")

# --- DERS 3: MORFOLOJİK İŞLEMLER ---
elif app_mode == "3. Morfolojik İşlemler (Noise Removal)":
    st.sidebar.subheader("Ayarlar")
    morph_type = st.sidebar.radio("İşlem:", ["Erosion (Aşındırma)", "Dilation (Genişletme)"])
    kernel_size = st.sidebar.slider("Kernel Boyutu", 1, 10, 3)
    iterations = st.sidebar.slider("Tekrar Sayısı", 1, 5, 1)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if morph_type == "Erosion (Aşındırma)":
        # Beyaz bölgeleri küçültür (Gürültü yok etmede kullanılır)
        processed_image = cv2.erode(original_image, kernel, iterations=iterations)
        desc = "Nesnelerin sınırlarını aşındırır ve küçültür. Küçük beyaz gürültüleri yok eder."
    else:
        # Beyaz bölgeleri büyütür (Kopuk parçaları birleştirmede kullanılır)
        processed_image = cv2.dilate(original_image, kernel, iterations=iterations)
        desc = "Nesnelerin sınırlarını genişletir. Kopuk çizgileri birleştirmek için kullanılır."
        
    with col2:
        st.subheader("Sonuç")
        st.image(processed_image, use_container_width=True)
        st.info(desc)

# --- DERS 4: YÜZ VE GÖZ TESPİTİ ---
elif app_mode == "4. Yüz ve Göz Tespiti (Object Detection)":
    st.sidebar.subheader("Algılama Ayarları")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.50, 1.1)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5)
    
    # Haar Cascade Dosyalarını Yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
    # Yüzleri Bul
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    
    # Çizim Yapacağımız Kopya
    img_out = original_image.copy()
    
    eye_count = 0
    for (x, y, w, h) in faces:
        # Yüze çerçeve çiz (Mavi)
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # Gözleri sadece yüzün içinde ara (Performans için)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_out[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            # Gözlere çerçeve çiz (Yeşil)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_count += 1
            
    with col2:
        st.subheader(f"Tespit Edilenler")
        st.image(img_out, use_container_width=True)
        st.success(f"Yüz Sayısı: {len(faces)} | Göz Sayısı: {eye_count}")
        st.caption("Mavi Kutu: Yüzler | Yeşil Kutu: Gözler")