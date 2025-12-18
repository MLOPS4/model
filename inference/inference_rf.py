import joblib
import pandas as pd
import numpy as np

class RFPredictor:
    def __init__(self, model_path: str):
        package = joblib.load(model_path)
        self.model = package['model_rf']
        self.le_prov = package['le_prov']
        self.le_kom = package['le_kom']
        self.data_ref = package['data_ref']
        
        self.mapping_bulan = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
            7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
    
    def get_last_data(self, provinsi: str, komoditas: str):
        """Ambil data terakhir untuk provinsi dan komoditas tertentu"""
        mask = (self.data_ref['Nama Provinsi'] == provinsi) & (self.data_ref['Komoditas'] == komoditas)
        subset = self.data_ref[mask].sort_values(by=['Tahun', 'Bulan_Angka'])
        
        if subset.empty:
            raise ValueError(f"Data tidak ditemukan untuk {provinsi} - {komoditas}")
        
        return subset.iloc[-1]
    
    def predict(self, provinsi: str, komoditas: str, bulan_target: int) -> dict:
        """
        Prediksi harga untuk bulan tertentu
        
        Args:
            provinsi: Nama provinsi (e.g., "Aceh")
            komoditas: Nama komoditas (e.g., "Bawang Merah")
            bulan_target: Jumlah bulan ke depan untuk diprediksi (1-12)
        
        Returns:
            dict dengan hasil prediksi
        """
        # Ambil data terakhir
        last_row = self.get_last_data(provinsi, komoditas)
        
        # Encode provinsi dan komoditas
        try:
            prov_id = self.le_prov.transform([provinsi])[0]
            kom_id = self.le_kom.transform([komoditas])[0]
        except ValueError as e:
            raise ValueError(f"Provinsi atau komoditas tidak dikenal: {e}")
        
        # Hitung bulan prediksi
        bulan_terakhir = int(last_row['Bulan_Angka'])
        tahun_terakhir = int(last_row['Tahun'])
        
        bulan_prediksi = (bulan_terakhir + bulan_target - 1) % 12 + 1
        tahun_prediksi = tahun_terakhir + (bulan_terakhir + bulan_target - 1) // 12
        
        # Siapkan fitur untuk prediksi iteratif
        harga_bulan_lalu = last_row['Harga']
        harga_tahun_lalu = last_row['Harga_Tahun_Lalu']
        
        # Prediksi iteratif bulan per bulan
        current_bulan = bulan_terakhir
        for _ in range(bulan_target):
            current_bulan = current_bulan % 12 + 1
            
            features = pd.DataFrame([[
                harga_bulan_lalu,
                harga_tahun_lalu,
                current_bulan,
                prov_id,
                kom_id
            ]], columns=['Harga_Bulan_Lalu', 'Harga_Tahun_Lalu', 'Bulan_Angka', 'Provinsi_ID', 'Komoditas_ID'])
            
            harga_prediksi = self.model.predict(features)[0]
            
            # Update untuk iterasi berikutnya
            harga_tahun_lalu = harga_bulan_lalu
            harga_bulan_lalu = harga_prediksi
        
        return {
            'lokasi': provinsi,
            'komoditas': komoditas,
            'dataTerakhir': {
                'bulan': self.mapping_bulan[bulan_terakhir],
                'tahun': tahun_terakhir,
                'harga': float(last_row['Harga'])
            },
            'prediksi': {
                'bulanKe': self.mapping_bulan[bulan_prediksi],
                'tahun': tahun_prediksi,
                'harga': float(harga_prediksi)
            }
        }


if __name__ == "__main__":
    # Contoh penggunaan
    predictor = RFPredictor('../train/model_v1.pkl')
    
    result = predictor.predict(
        provinsi="Aceh",
        komoditas="Bawang Merah",
        bulan_target=7
    )
    
    print(f"Lokasi: {result['lokasi']}")
    print(f"Komoditas: {result['komoditas']}")
    print(f"Data Terakhir: {result['dataTerakhir']['bulan']} {result['dataTerakhir']['tahun']} - Rp {result['dataTerakhir']['harga']:,.0f}")
    print(f"Prediksi: {result['prediksi']['bulanKe']} {result['prediksi']['tahun']} - Rp {result['prediksi']['harga']:,.0f}")

