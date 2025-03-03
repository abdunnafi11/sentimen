from sklearn.utils.validation import check_is_fitted
import streamlit as st
import pickle
import time

# Load model dan vectorizer
model = pickle.load(open('sentiment_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf_new.pkl', 'rb'))

st.title('📊 Sentiment Analysis Pengguna Aplikasi Lazada')

coms = st.text_area('📝 Masukkan Review Anda Tentang Aplikasi Kami')

if st.button('🔍 Prediksi'):
    if not coms.strip():
        st.warning("⚠️ Silakan masukkan review terlebih dahulu.")
    else:
        try:
            # Cek apakah vectorizer sudah dilatih
            from sklearn.utils.validation import check_is_fitted

try:
    check_is_fitted(vectorizer, attributes=["idf_"])
except:
    raise ValueError("❌ Vectorizer belum dilatih! Pastikan Anda menggunakan file vectorizer yang benar.")

            
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
            st.error(f"⚠️ Error: {str(e)}. Coba muat ulang model atau latih ulang vectorizer.")
