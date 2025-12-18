import joblib
import pandas as pd
from dateutil.relativedelta import relativedelta

class ARIMAPredictor:
    def __init__(self, model_path: str):
        package = joblib.load(model_path)
        self.arima_models = package['arima_models']
        self.mapping_bulan = package['mapping_bulan']
        self.data_ref = package['data_ref']
        
        # Reverse mapping: angka -> nama bulan
        self.bulan_to_name = {v: k for k, v in self.mapping_bulan.items()}
    
    def get_available_models(self) -> list:
        """Daftar kombinasi (provinsi, komoditas) yang tersedia"""
        return list(self.arima_models.keys())
    
    def get_last_data(self, provinsi: str, komoditas: str) -> dict:
        """Ambil data terakhir untuk provinsi dan komoditas tertentu"""
        mask = (self.data_ref['Nama Provinsi'] == provinsi) & (self.data_ref['Komoditas'] == komoditas)
        subset = self.data_ref[mask].sort_values(by=['Tahun', 'Bulan_Angka'])
        
        if subset.empty:
            raise ValueError(f"Data tidak ditemukan untuk {provinsi} - {komoditas}")
        
        last_row = subset.iloc[-1]
        return {
            'bulan': self.bulan_to_name[int(last_row['Bulan_Angka'])],
            'tahun': int(last_row['Tahun']),
            'harga': float(last_row['Harga'])
        }
    
    def predict(self, provinsi: str, komoditas: str, bulan_target: int) -> dict:
        """
        Prediksi harga untuk bulan tertentu menggunakan ARIMA
        
        Args:
            provinsi: Nama provinsi (e.g., "Aceh")
            komoditas: Nama komoditas (e.g., "Bawang Merah")
            bulan_target: Jumlah bulan ke depan untuk diprediksi (1-24)
        
        Returns:
            dict dengan hasil prediksi
        """
        key = (provinsi, komoditas)
        
        if key not in self.arima_models:
            available = [f"{p} - {k}" for p, k in list(self.arima_models.keys())[:5]]
            raise ValueError(f"Model tidak ditemukan untuk {provinsi} - {komoditas}. "
                           f"Contoh yang tersedia: {available}")
        
        model_data = self.arima_models[key]
        fitted_model = model_data['model']
        last_date = model_data['last_date']
        
        # Forecast untuk bulan_target ke depan
        forecast = fitted_model.forecast(steps=bulan_target)
        harga_prediksi = forecast.iloc[-1]  # Ambil prediksi terakhir
        
        # Hitung tanggal prediksi
        target_date = last_date + relativedelta(months=bulan_target)
        bulan_prediksi = target_date.month
        tahun_prediksi = target_date.year
        
        # Ambil data terakhir dari referensi
        last_data = self.get_last_data(provinsi, komoditas)
        
        return {
            'lokasi': provinsi,
            'komoditas': komoditas,
            'dataTerakhir': last_data,
            'prediksi': {
                'bulanKe': self.bulan_to_name[bulan_prediksi],
                'tahun': tahun_prediksi,
                'harga': float(harga_prediksi)
            }
        }
    
    def predict_range(self, provinsi: str, komoditas: str, bulan_target: int) -> dict:
        """
        Prediksi harga untuk rentang bulan (semua bulan dari 1 sampai bulan_target)
        
        Returns:
            dict dengan list prediksi per bulan
        """
        key = (provinsi, komoditas)
        
        if key not in self.arima_models:
            raise ValueError(f"Model tidak ditemukan untuk {provinsi} - {komoditas}")
        
        model_data = self.arima_models[key]
        fitted_model = model_data['model']
        last_date = model_data['last_date']
        
        # Forecast semua bulan
        forecast = fitted_model.forecast(steps=bulan_target)
        
        predictions = []
        for i, harga in enumerate(forecast, start=1):
            target_date = last_date + relativedelta(months=i)
            predictions.append({
                'bulan': self.bulan_to_name[target_date.month],
                'tahun': target_date.year,
                'harga': float(harga)
            })
        
        last_data = self.get_last_data(provinsi, komoditas)
        
        return {
            'lokasi': provinsi,
            'komoditas': komoditas,
            'dataTerakhir': last_data,
            'prediksi': predictions
        }


if __name__ == "__main__":
    # Contoh penggunaan
    predictor = ARIMAPredictor('../train/model_arima_v1.pkl')
    
    # Lihat model yang tersedia
    print(f"Model tersedia: {len(predictor.get_available_models())} kombinasi\n")
    
    # Prediksi single
    result = predictor.predict(
        provinsi="Aceh",
        komoditas="Bawang Merah",
        bulan_target=7
    )
    
    print(f"Lokasi: {result['lokasi']}")
    print(f"Komoditas: {result['komoditas']}")
    print(f"Data Terakhir: {result['dataTerakhir']['bulan']} {result['dataTerakhir']['tahun']} - Rp {result['dataTerakhir']['harga']:,.0f}")
    print(f"Prediksi: {result['prediksi']['bulanKe']} {result['prediksi']['tahun']} - Rp {result['prediksi']['harga']:,.0f}")
    
    # Prediksi range (semua bulan)
    print("\n--- Prediksi Range ---")
    result_range = predictor.predict_range(
        provinsi="Aceh",
        komoditas="Bawang Merah",
        bulan_target=6
    )
    for pred in result_range['prediksi']:
        print(f"  {pred['bulan']} {pred['tahun']}: Rp {pred['harga']:,.0f}")

