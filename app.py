from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted
import streamlit as st
import pickle
import time

# Load model dan CountVectorizer
model = pickle.load(open('sentiment_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_count.pkl', 'rb'))  # Ganti dengan CountVectorizer yang sudah dilatih

st.title('📊 Sentiment Analysis Pengguna Aplikasi Lazada')

coms = st.text_area('📝 Masukkan Review Anda Tentang Aplikasi Kami')

if st.button('🔍 Prediksi'):
    if not coms.strip():
        st.warning("⚠️ Silakan masukkan review terlebih dahulu.")
    else:
        try:
            # Cek apakah vectorizer sudah dilatih
            check_is_fitted(vectorizer, attributes=["vocabulary_"])

            with st.spinner("🔄 Menganalisis sentimen..."):
                start = time.time()
                transformed_text = vectorizer.transform([coms])
                prediction = model.predict(transformed_text)
                end = time.time()
                st.write(f'⏱️ Waktu prediksi: {round(end-start, 2)} detik')

                if prediction[0] == 1:
                    st.success("😊 Sentimen review Anda *positif*!")
                else:
                    st.error("😞 Sentimen review Anda *negatif*.")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}. Pastikan model dan vectorizer telah dilatih dengan benar.")
