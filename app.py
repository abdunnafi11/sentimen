import streamlit as st
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = pickle.load(open('sentiment_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf_new.pkl', 'rb'))

# Tampilan judul
st.title('📊 Sentiment Analysis Pengguna Aplikasi Lazada')

# Input teks
coms = st.text_area('📝 Masukkan Review Anda Tentang Aplikasi Kami')

# Tombol prediksi
if st.button('🔍 Prediksi'):
    if not coms.strip():
        st.warning("⚠️ Silakan masukkan review terlebih dahulu.")
    else:
        with st.spinner("🔄 Menganalisis sentimen..."):
            start = time.time()

            # Transform teks input
            transformed_text = vectorizer.transform([coms])

            # Prediksi sentimen
            prediction = model.predict(transformed_text)

            end = time.time()
            st.write(f'⏱️ Waktu prediksi: {round(end-start, 2)} detik')

            # Output hasil prediksi
            if prediction[0] == 1:
                st.success("😊 Sentimen review Anda *positif*!")
            else:
                st.error("😞 Sentimen review Anda *negatif*.")
