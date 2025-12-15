import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Library Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Membaca file
df = pd.read_csv('./dataset_clean/1763087087.csv_clean.csv', skiprows=2)

# Membersihkan nama kolom (menghapus spasi tersembunyi)
df.columns = df.columns.str.strip()

# Membersihkan format mata uang ("Rp" dan ",")
# Mengubah ke tipe data String dulu, baru replace
df['Harga'] = df['Harga'].astype(str).str.replace('Rp', '').str.replace(',', '')

# Mengubah ke tipe Numerik (Float)
# errors='coerce' mengubah data non-angka (seperti "-") menjadi NaN
df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')

# Mapping Nama Bulan ke Angka
df['Bulan'] = df['Bulan'].astype(str).str.strip()
mapping_bulan = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}
df['Bulan_Angka'] = df['Bulan'].map(mapping_bulan)

# Hapus baris yang gagal di-mapping (baris sampah)
df = df.dropna(subset=['Bulan_Angka'])

# Urutkan data agar interpolasi berjalan sesuai urutan waktu
df = df.sort_values(by=['Nama Provinsi', 'Komoditas', 'Tahun', 'Bulan_Angka'])

# Interpolasi Linear (Mengisi data bolong berdasarkan rata-rata tetangganya)
# Dilakukan per Grup (Provinsi & Komoditas) agar data tidak bocor antar daerah
df['Harga'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].transform(lambda x: x.interpolate())

# Hapus data yang masih kosong (biasanya data di bulan-bulan awal periode)
df = df.dropna(subset=['Harga'])

# Hapus data Harga 0 atau Negatif (Penyebab Error MAPE)
df = df[df['Harga'] > 0]

print("✅ Data Cleaning selesai.")
df = df.reset_index(drop=True)

# Membuat Fitur LAG (Harga Masa Lalu)
# Shift(1) = Harga 1 Bulan Lalu
# Shift(12) = Harga 1 Tahun Lalu (Seasonality)
df['Harga_Bulan_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(1)
df['Harga_3Bulan_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(3)
df['Harga_Tahun_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(12)

# Encoding Data Kategorikal (Mengubah Teks jadi Angka ID)
le_prov = LabelEncoder()
df['Provinsi_ID'] = le_prov.fit_transform(df['Nama Provinsi'])

le_kom = LabelEncoder()
df['Komoditas_ID'] = le_kom.fit_transform(df['Komoditas'])

# Hapus baris NaN yang muncul akibat proses Shift (Lagging)
df_clean = df.dropna()

print("✅ Feature Engineering selesai.")
# Definisi Fitur (X) dan Target (y)
features = ['Harga_Bulan_Lalu', 'Harga_Tahun_Lalu', 'Bulan_Angka', 'Provinsi_ID', 'Komoditas_ID']
target = 'Harga'

# Time Series Split (Potong berdasarkan Tahun)
# Training: Data sebelum 2024
# Testing: Data 2024 ke atas
X_train = df_clean[df_clean['Tahun'] < 2024][features]
y_train = df_clean[df_clean['Tahun'] < 2024][target]

X_test = df_clean[df_clean['Tahun'] >= 2024][features]
y_test = df_clean[df_clean['Tahun'] >= 2024][target]

# Inisialisasi & Latih Model Random Forest
# model = RandomForestRegressor(n_estimators=200, random_state=42)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

model.fit(X_train, y_train)

print(f"✅ Model berhasil dilatih! (Train: {len(X_train)} baris, Test: {len(X_test)} baris)")