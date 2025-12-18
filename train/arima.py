import pandas as pd
import numpy as np
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# Membaca file
df = pd.read_csv('../dataset_clean/1763087087.csv_clean.csv')

# Membersihkan nama kolom (menghapus spasi tersembunyi)
df.columns = df.columns.str.strip()

# Membersihkan format mata uang ("Rp" dan ",")
df['Harga'] = df['Harga'].astype(str).str.replace('Rp', '').str.replace(',', '')

# Mengubah ke tipe Numerik (Float)
df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')

# Mapping Nama Bulan ke Angka
df['Bulan'] = df['Bulan'].astype(str).str.strip()
mapping_bulan = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}
df['Bulan_Angka'] = df['Bulan'].map(mapping_bulan)

# Hapus baris yang gagal di-mapping
df = df.dropna(subset=['Bulan_Angka'])

# Urutkan data berdasarkan waktu
df = df.sort_values(by=['Nama Provinsi', 'Komoditas', 'Tahun', 'Bulan_Angka'])

# Interpolasi Linear per Grup
df['Harga'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].transform(lambda x: x.interpolate())

# Hapus data kosong dan harga <= 0
df = df.dropna(subset=['Harga'])
df = df[df['Harga'] > 0]
df = df.reset_index(drop=True)

print("✅ Data Cleaning selesai.")

# Buat kolom tanggal untuk time series index
df['Tanggal'] = pd.to_datetime(df['Tahun'].astype(int).astype(str) + '-' + df['Bulan_Angka'].astype(int).astype(str) + '-01')

# Dictionary untuk menyimpan semua model ARIMA
arima_models = {}
results_summary = []

# Grup unik (Provinsi, Komoditas)
groups = df.groupby(['Nama Provinsi', 'Komoditas'])

print(f"📊 Training ARIMA untuk {len(groups)} kombinasi Provinsi-Komoditas...\n")

for (provinsi, komoditas), group_df in groups:
    # Siapkan time series
    ts = group_df.set_index('Tanggal')['Harga'].sort_index()
    
    # Skip jika data terlalu sedikit (minimal 24 bulan untuk ARIMA)
    if len(ts) < 24:
        print(f"⚠️  Skip {provinsi} - {komoditas}: Data kurang ({len(ts)} bulan)")
        continue
    
    # Split train/test (data sebelum 2024 untuk training)
    train = ts[ts.index < '2024-01-01']
    test = ts[ts.index >= '2024-01-01']
    
    if len(train) < 12:
        print(f"⚠️  Skip {provinsi} - {komoditas}: Training data kurang")
        continue
    
    try:
        # Fit ARIMA model (p=1, d=1, q=1) - parameter umum untuk data harga
        # Bisa diganti dengan auto_arima dari pmdarima untuk optimasi otomatis
        model = ARIMA(train, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Simpan model
        key = (provinsi, komoditas)
        arima_models[key] = {
            'model': fitted_model,
            'last_data': train.iloc[-12:].values,  # 12 bulan terakhir untuk referensi
            'last_date': train.index[-1]
        }
        
        # Evaluasi jika ada test data
        if len(test) > 0:
            forecast = fitted_model.forecast(steps=len(test))
            mae = mean_absolute_error(test, forecast)
            mape = mean_absolute_percentage_error(test, forecast) * 100
            results_summary.append({
                'Provinsi': provinsi,
                'Komoditas': komoditas,
                'MAE': mae,
                'MAPE': mape,
                'Train_Size': len(train),
                'Test_Size': len(test)
            })
            print(f"✅ {provinsi} - {komoditas}: MAE={mae:.2f}, MAPE={mape:.2f}%")
        else:
            print(f"✅ {provinsi} - {komoditas}: Model trained (no test data)")
            
    except Exception as e:
        print(f"❌ Error {provinsi} - {komoditas}: {e}")
        continue

print(f"\n📈 Total model berhasil: {len(arima_models)}")

# Ringkasan hasil evaluasi
if results_summary:
    results_df = pd.DataFrame(results_summary)
    print(f"\n📊 Rata-rata MAPE: {results_df['MAPE'].mean():.2f}%")
    print(f"📊 Rata-rata MAE: {results_df['MAE'].mean():.2f}")

# Simpan model package
model_package = {
    'arima_models': arima_models,
    'mapping_bulan': mapping_bulan,
    'data_ref': df
}

joblib.dump(model_package, 'model_arima_v1.pkl')
print("\n✅ Model ARIMA berhasil disimpan ke model_arima_v1.pkl")

