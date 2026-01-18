from pathlib import Path
import pandas as pd

cleaned_path = Path('data/processed/cleaned.parquet')
df = pd.read_parquet(cleaned_path)
print(f'Shape TRƯỚC lọc: {df.shape}')
print(f'  Dòng: {df.shape[0]:,}')
print(f'  Cột: {df.shape[1]}')
print()

before = len(df)
df_filtered = df[df['aqi_class'].notna()].copy()
print(f'Shape SAU lọc: {df_filtered.shape}')
print(f'  Dòng: {df_filtered.shape[0]:,}')
print(f'  Đã loại: {before - len(df_filtered):,} dòng (aqi_class = NaN)')
print()

print('=' * 80)
print(f'TẤT CẢ CÁC CỘT SAU NOTEBOOK 2: ({len(df_filtered.columns)} cột)')
print('=' * 80)
for i, col in enumerate(df_filtered.columns, 1):
    print(f'{i:2}. {col}')

print()
print('=' * 80)
print('FEATURES CHO MODELING (loại PM2.5, pm25_24h, aqi_class, datetime):')
print('=' * 80)
drop_cols = {'PM2.5', 'pm25_24h', 'aqi_class', 'datetime'}
feature_cols = [c for c in df_filtered.columns if c not in drop_cols]
print(f'Số features: {len(feature_cols)} cột')
print()
for i, col in enumerate(feature_cols, 1):
    print(f'{i:2}. {col}')

print()
print('=' * 80)
print('MẪU DỮ LIỆU (10 dòng đầu):')
print('=' * 80)
sample_cols = ['datetime', 'station', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 
               'TEMP', 'PRES', 'hour', 'day', 'month', 'is_weekend', 
               'PM2.5_lag_1h', 'PM2.5_lag_3h', 'pm25_24h', 'aqi_class']
available = [c for c in sample_cols if c in df_filtered.columns]
print(df_filtered[available].head(10))

print()
print('=' * 80)
print('PHÂN BỐ AQI_CLASS:')
print('=' * 80)
print(df_filtered['aqi_class'].value_counts().sort_values(ascending=False))
