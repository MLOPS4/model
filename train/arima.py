import pandas as pd
import numpy as np
import joblib
import warnings
import mlflow
import mlflow.sklearn
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# MLflow setup
mlflow.set_experiment("arima-harga-pangan")

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

# ARIMA parameters
ARIMA_ORDER = (1, 1, 1)
TRAIN_TEST_SPLIT_YEAR = 2024
MIN_DATA_POINTS = 24
MIN_TRAIN_POINTS = 12

with mlflow.start_run(run_name="arima-training"):
    # Log parameters
    mlflow.log_param("arima_order_p", ARIMA_ORDER[0])
    mlflow.log_param("arima_order_d", ARIMA_ORDER[1])
    mlflow.log_param("arima_order_q", ARIMA_ORDER[2])
    mlflow.log_param("train_test_split_year", TRAIN_TEST_SPLIT_YEAR)
    mlflow.log_param("min_data_points", MIN_DATA_POINTS)
    mlflow.log_param("min_train_points", MIN_TRAIN_POINTS)
    mlflow.log_param("dataset", "1763087087.csv_clean.csv")
    
    # Dictionary untuk menyimpan semua model ARIMA
    arima_models = {}
    results_summary = []
    
    # Grup unik (Provinsi, Komoditas)
    groups = df.groupby(['Nama Provinsi', 'Komoditas'])
    total_groups = len(groups)
    
    mlflow.log_param("total_groups", total_groups)
    print(f"📊 Training ARIMA untuk {total_groups} kombinasi Provinsi-Komoditas...\n")
    
    skipped_count = 0
    error_count = 0
    
    for (provinsi, komoditas), group_df in groups:
        # Siapkan time series
        ts = group_df.set_index('Tanggal')['Harga'].sort_index()
        
        # Skip jika data terlalu sedikit (minimal 24 bulan untuk ARIMA)
        if len(ts) < MIN_DATA_POINTS:
            print(f"⚠️  Skip {provinsi} - {komoditas}: Data kurang ({len(ts)} bulan)")
            skipped_count += 1
            continue
        
        # Split train/test (data sebelum 2024 untuk training)
        train = ts[ts.index < f'{TRAIN_TEST_SPLIT_YEAR}-01-01']
        test = ts[ts.index >= f'{TRAIN_TEST_SPLIT_YEAR}-01-01']
        
        if len(train) < MIN_TRAIN_POINTS:
            print(f"⚠️  Skip {provinsi} - {komoditas}: Training data kurang")
            skipped_count += 1
            continue
        
        try:
            # Fit ARIMA model
            model = ARIMA(train, order=ARIMA_ORDER)
            fitted_model = model.fit()
            
            # Simpan model
            key = (provinsi, komoditas)
            arima_models[key] = {
                'model': fitted_model,
                'last_data': train.iloc[-12:].values,
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
            error_count += 1
            continue
    
    # Log training summary metrics
    mlflow.log_metric("total_models_trained", len(arima_models))
    mlflow.log_metric("skipped_count", skipped_count)
    mlflow.log_metric("error_count", error_count)
    
    print(f"\n📈 Total model berhasil: {len(arima_models)}")
    
    # Ringkasan hasil evaluasi
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        avg_mape = results_df['MAPE'].mean()
        avg_mae = results_df['MAE'].mean()
        min_mape = results_df['MAPE'].min()
        max_mape = results_df['MAPE'].max()
        
        # Log aggregate metrics
        mlflow.log_metric("avg_mape", avg_mape)
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("min_mape", min_mape)
        mlflow.log_metric("max_mape", max_mape)
        mlflow.log_metric("evaluated_models", len(results_summary))
        
        print(f"\n📊 Rata-rata MAPE: {avg_mape:.2f}%")
        print(f"📊 Rata-rata MAE: {avg_mae:.2f}")
        
        # Save results summary as artifact
        results_df.to_csv('arima_results_summary.csv', index=False)
        mlflow.log_artifact('arima_results_summary.csv')
    
    # Simpan model package
    model_package = {
        'arima_models': arima_models,
        'mapping_bulan': mapping_bulan,
        'data_ref': df
    }
    
    model_path = 'model_arima_v1.pkl'
    joblib.dump(model_package, model_path)
    mlflow.log_artifact(model_path)
    
    # Log model info as tags
    mlflow.set_tag("model_type", "ARIMA")
    mlflow.set_tag("framework", "statsmodels")
    
    print("\n✅ Model ARIMA berhasil disimpan ke model_arima_v1.pkl")
    print(f"✅ MLflow run logged: {mlflow.active_run().info.run_id}")

