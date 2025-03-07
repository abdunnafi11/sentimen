import streamlit as st
import pickle
import time

# Load model dan vectorizer
try:
    model = pickle.load(open('sentiment.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))
except Exception as e:
    st.error(f"Error saat memuat model atau vectorizer: {e}")
    st.stop()

st.title('Sentiment Analysis pengguna Aplikasi Lazada')

# Input dari user
coms = st.text_input('Masukkan Review Anda Tentang Aplikasi Kami')

submit = st.button('Prediksi')

if submit:
    start = time.time()

    # Cek apakah vectorizer valid
    if vectorizer is None:
        st.error("Vectorizer tidak dimuat dengan benar!")
        st.stop()

    # Cek apakah input kosong
    if not coms.strip():
        st.warning("Masukkan teks sebelum melakukan prediksi!")
        st.stop()

    # Transformasi teks
    try:
        transformed_text = vectorizer.transform([coms])  # Hapus .toarray()
        prediction = model.predict(transformed_text)

        end = time.time()
        st.write('Waktu prediksi: ', round(end - start, 2), 'detik')

        if prediction[0] == 1:
            st.success("Sentimen review Anda positif 😊")
        else:
            st.error("Sentimen review Anda negatif 😞")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat transformasi atau prediksi: {e}")
