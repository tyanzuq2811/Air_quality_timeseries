"""
Script đơn giản để chạy toàn bộ notebook preprocessing_and_eda.ipynb
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification_library import (
    load_beijing_air_quality,
    clean_air_quality_df,
    add_pm25_24h_and_label,
    add_time_features,
    add_lag_features,
)

# Parameters
USE_UCIMLREPO = False
RAW_ZIP_PATH = str(PROJECT_ROOT / "data/raw/PRSA2017_Data_20130301-20170228.zip")
OUTPUT_CLEANED_PATH = 'data/processed/cleaned.parquet'
LAG_HOURS = [1, 3, 24]

# Load and process data
print("Loading raw data...")
df_raw = load_beijing_air_quality(use_ucimlrepo=USE_UCIMLREPO, raw_zip_path=RAW_ZIP_PATH)
print(f'Raw shape: {df_raw.shape}')

print("\nCleaning and creating features...")
df = clean_air_quality_df(df_raw)
df = add_pm25_24h_and_label(df)
df = add_time_features(df)
df = add_lag_features(df, lag_hours=LAG_HOURS)
print(f'Cleaned shape: {df.shape}')

print("\n" + "="*80)
print("EDA ANALYSIS COMPLETED")
print("="*80)
print(f"\nTotal samples: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Stations: {df['station'].nunique()}")
print(f"\nMissing PM2.5: {df['PM2.5'].isna().mean()*100:.2f}%")
print(f"Valid target (aqi_class): {df['aqi_class'].notna().sum():,}")

# Save
OUT_PATH = (PROJECT_ROOT / OUTPUT_CLEANED_PATH).resolve()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')
print("\nNotebook 1 processing completed!")
