# Rain Tomorrow — Project Package

## Isi folder
- `rain-tomorrow-leakage-safe-explainable(1).ipynb`: notebook EDA, preprocessing, eksperimen model dan evaluasi (unggah ke Google Colab).
- `app.py`: aplikasi Streamlit untuk prediksi data baru.
- `requirements.txt`: dependensi.
- `weatherAUS.csv`: letakkan dataset CSV ini di folder yang sama sebelum menjalankan aplikasi.
- `slides_presentasi.pptx`: bahan presentasi awal.

## Dataset
Unduh `weatherAUS.csv` dari Kaggle: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
Simpan di folder ini dengan nama persis `weatherAUS.csv`. Notebook yang disediakan mengunduh dataset melalui `kagglehub` saat dijalankan di Colab.

## Google Colab
1. Buka https://colab.research.google.com/
2. Upload file notebook `.ipynb`.
3. Jalankan semua sel secara berurutan. Notebook mengunduh dataset melalui `kagglehub`.
4. Setelah tersimpan di Drive, salin link Colab dan ubah izin akses menjadi siapa pun yang memiliki link dapat melihat.
5. Tempel link tersebut di slide presentasi.

## Jalankan Streamlit secara lokal
1. Letakkan `weatherAUS.csv` di folder yang sama dengan `app.py`.
2. Buat environment Python (disarankan Python 3.10/3.11).
3. Jalankan:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Buka URL lokal yang ditampilkan di terminal.

## Catatan metodologi penting
Notebook yang dilampirkan memakai split kronologis train/validation/test dan membandingkan baseline, Logistic Regression, dan LightGBM. Jangan gunakan `RISK_MM` sebagai fitur karena fitur itu berhubungan langsung dengan target `RainTomorrow` dan menimbulkan target leakage.
Aplikasi demo melatih Logistic Regression dengan holdout acak agar mudah dijalankan. Untuk angka hasil eksperimen yang dilaporkan, gunakan evaluasi kronologis dari notebook, bukan angka holdout aplikasi.
Dataset tidak disertakan otomatis karena unduhan CSV eksternal tidak berhasil pada sesi pembuatan paket ini; unduh dari tautan Kaggle di atas.
