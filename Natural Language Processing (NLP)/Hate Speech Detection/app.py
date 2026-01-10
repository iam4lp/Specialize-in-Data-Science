# app.py
import streamlit as st
import joblib
import re
import string

# Modeli Yükle
try:
    model = joblib.load('src/hate_speech_model.pkl')
except:
    st.error("Model dosyası bulunamadı. Lütfen 'train_model.py' dosyasını çalıştırın.")
    st.stop()

# Temizleme Fonksiyonu (Eğitimdeki ile aynı mantıkta olmalı)
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Sayfa Tasarımı
st.set_page_config(page_title="Hate Speech Detection", page_icon="🚫")

st.title("🚫 Hate Speech Detection")
st.markdown("Bu uygulama, girilen metnin **Nefret Söylemi**, **Saldırgan Dil** veya **Temiz** olup olmadığını tespit eder.")
st.info("Not: Model İngilizce tweet veri seti üzerinde eğitilmiştir.")

# Kullanıcı Girişi
user_input = st.text_area("Analiz edilecek metni giriniz:", height=100, placeholder="Type something here...")

if st.button("Analiz Et"):
    if user_input:
        # Metni temizle
        cleaned_input = clean_text(user_input)
        
        # Tahmin yap
        prediction = model.predict([cleaned_input])[0]
        
        # Sonucu Ekrana Bas
        # Class 0: Hate Speech, 1: Offensive Language, 2: Neither
        
        if prediction == 0:
            st.error("SONUÇ: 🤬 Hate Speech (Nefret Söylemi) Tespit Edildi!")
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2Q4NzE4.../giphy.gif", width=100) # Opsiyonel görsel
        elif prediction == 1:
            st.warning("SONUÇ: 😡 Offensive Language (Saldırgan Dil)")
        else:
            st.success("SONUÇ: ✅ Neither (Temiz / Nötr)")
            
    else:
        st.write("Lütfen bir metin giriniz.")