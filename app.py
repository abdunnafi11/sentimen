import streamlit as st
import pickle
import numpy as np
import time
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = pickle.load(open('sentiment_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf_new.pkl', 'rb'))

# Konfigurasi tampilan Streamlit
def set_theme():
    st.markdown(
        """
        <style>
        body {background-color: #f5f7fa;}
        .main {background-color: white; padding: 20px; border-radius: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_theme()

# Judul utama dengan ikon
st.title("📊 Sentiment Analysis Pengguna Aplikasi Lazada")
st.markdown("**Analisis sentimen review pengguna menggunakan Machine Learning.**")

# Sidebar untuk informasi tambahan
st.sidebar.header("Informasi")
st.sidebar.write("Aplikasi ini menggunakan **TF-IDF Vectorizer** dan **Machine Learning** untuk menganalisis sentimen review pengguna.")

# Input pengguna dengan tampilan lebih menarik
coms = st.text_area("💬 Masukkan Review Anda tentang Aplikasi Kami", "", height=150)
submit = st.button("🔍 Prediksi Sentimen")

if submit:
    with st.spinner("Sedang menganalisis sentimen..."):
        start = time.time()
        transformed_text = vectorizer.transform([coms]).toarray()
        prediction = model.predict(transformed_text)
        end = time.time()
        st.success(f"✅ Analisis selesai dalam {round(end-start, 2)} detik")

        # Hasil prediksi dengan ikon
        if prediction[0] == 1:
            st.markdown("<h3 style='color:green;'>😊 Sentimen review Anda POSITIF</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>😞 Sentimen review Anda NEGATIF</h3>", unsafe_allow_html=True)

        # Simulasi data visualisasi
        labels = ['Positif', 'Negatif']
        values = [np.random.randint(30, 70), np.random.randint(10, 40)]

        # Visualisasi pie chart
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        ax.set_title("Distribusi Sentimen Pengguna")
        st.pyplot(fig)

# Footer aplikasi
st.markdown("""
---
👨‍💻 *Dibuat dengan ❤️ oleh Tim Data Science*
""")
