import streamlit as st
import pickle
import numpy as np
import time
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = pickle.load(open('sentiment.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf_cobalagi.pkl', 'rb'))

# Cek tipe objek vectorizer
if not isinstance(vectorizer, TfidfVectorizer):
    st.error("File vectorizer yang dimuat bukan TfidfVectorizer!")
    st.stop()

st.title('Sentiment Analysis pengguna Aplikasi Lazada')
coms = st.text_input('Masukkan Review Anda Tentang Aplikasi Kami')

submit = st.button('Prediksi')

if submit:
    start = time.time()
    try:
        # Transformasi teks
        transformed_text = vectorizer.transform([coms])  # Pastikan .transform dipanggil
        prediction = model.predict(transformed_text)
        end = time.time()
        st.write('Waktu prediksi: ', round(end-start, 2), 'detik')

        if prediction[0] == 1:
            st.success("Sentimen review Anda positif 😊")
        else:
            st.error("Sentimen review Anda negatif 😞")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat transformasi atau prediksi: {e}")
