import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Open file
def clean_data(filename):
    df = pd.read_csv(f'./raw/{filename}', skiprows=2)
    df.columns = df.columns.str.strip()
    df['Harga'] = df['Harga'].astype(str).str.replace('Rp', '').str.replace(',', '')
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
    df['Bulan'] = df['Bulan'].astype(str).str.strip()
    mapping_bulan = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    df['Bulan_Angka'] = df['Bulan'].map(mapping_bulan)
    df = df.dropna(subset=['Bulan_Angka'])

    # Definisi Fitur (X) dan Target (y)
    df = df.reset_index(drop=True)
    df['Harga_Bulan_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(1)
    df['Harga_3Bulan_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(3)
    df['Harga_Tahun_Lalu'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga'].shift(12)

    le_prov = LabelEncoder()
    df['Provinsi_ID'] = le_prov.fit_transform(df['Nama Provinsi'])
    le_kom = LabelEncoder()
    df['Komoditas_ID'] = le_kom.fit_transform(df['Komoditas'])
    df_clean = df.dropna()

    df_clean.to_csv(f'./dataset/{filename}_clean.csv', index=False)

    return f"./dataset/{filename}_clean.csv"