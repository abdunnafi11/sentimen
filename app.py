

import streamlit as st
import pickle
import pandas as pd
import numpy as np
# import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from PIL import Image

# Load model dan vectorizer
model = pickle.load(open('sentiment_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf_new.pkl', 'rb'))

# Tampilan utama
st.set_page_config(page_title='Sentiment Analysis Lazada', page_icon='💬')
st.title('🛒 Sentiment Analysis Pengguna Aplikasi Lazada')
st.markdown("---")
if hasattr(vectorizer, "idf_"):
    print("✅ Vectorizer sudah di-fit.")
else:
    print("❌ Vectorizer belum di-fit!")

# Input review pengguna
st.subheader("📢 Masukkan Review Anda")
coms = st.text_area("Tulis ulasan tentang aplikasi kami:")

# Tombol prediksi
if st.button("🔍 Prediksi Sentimen"):
    if coms.strip():
        start = time.time()
        transformed_text = vectorizer.transform([coms]).toarray()
        transformed_text = transformed_text.reshape(1, -1)
        prediction = model.predict(transformed_text)
        end = time.time()
        
        st.markdown("---")
        st.write(f"⏳ Waktu prediksi: {round(end-start, 2)} detik")
        
        if prediction[0] == 1:
            st.success("✅ Sentimen review Anda **positif**! 😊")
        else:
            st.error("❌ Sentimen review Anda **negatif**. 😟")
    else:
        st.warning("⚠️ Harap masukkan teks review terlebih dahulu.")
