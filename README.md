# Tech stack
 
Parse document : Llamaparse

Model embedding : Pinecone using multilingual-e5-large

Model LLM : Open router using Deepseek

Deploy : Streamlit

# Development

```
pip install virtualenv
```
```
virtualenv venv
```
```
venv/Scripts/activate
```
```
pip install -r requirements.txt
```

Tunggu hingga proses install library selesai

## Ingest data

Masukkan semua document yang mau diingest ke folder documents, kemudian jalankan perintah dibawah ini untuk mengingest document ke dalam vector db.

```
python add-data.py
```

Buat file.env di folder project, copy file.env.example ke file.env. masukkin env yang dibutuhkan seperti openaikey, pinecone key, dan llama cloud key. Lalu terakhir jalankan perintah dibawah ini untuk menjalankan web streamlit secara lokal

```
streamlit run main.py
```

# Deploy

1. Buat akun streamlit dan login (pastikan login menggunakan akun Github)
https://streamlit.io/cloud
2. Di pojok kanan atas, klik "Create app" Lalu pilih "Deploy a public app from GitHub"
3. Pilih repository yang sesuai lalu klik "Deploy" 
![s09583704242025](https://a.okmd.dev/md/6809a8dce1075.png)
4. Tunggu hingga proses deploy selesai, kemudian website sudah bisa diakses


# FAQ
## How to add new documents
Ada dua cara untuk menambahkan dokumen baru, yaitu upload di page upload page streamlit dan ini memakan waktu yang lama.

Saya menyarankan upload melalui cara kedua yaitu dari development, karena support banyak dokumen sekaligus

## Supported documents types
https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types



```