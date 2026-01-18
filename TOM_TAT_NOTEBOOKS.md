# Tổng Hợp Notebooks - Dự Án Air Quality Time Series

> **Tài liệu tổng hợp**: Mô tả toàn bộ quy trình phân tích và mô hình hóa dữ liệu chất lượng không khí Beijing
> 
> **Ngày tạo**: 18/01/2026

---

## 📋 Tổng Quan Dự Án

Dự án phân tích và dự đoán chất lượng không khí (Air Quality) tại Beijing sử dụng dữ liệu từ nhiều trạm quan trắc từ 2013-2017. Dự án bao gồm 5 notebooks chính, thực hiện các bước từ tiền xử lý dữ liệu đến xây dựng các mô hình machine learning và time series forecasting.

**Dataset**: Beijing Multi-Site Air Quality Data (2013-2017)
- Nguồn: UCI Machine Learning Repository
- Dữ liệu đo lường theo giờ từ 12 trạm quan trắc
- Các thông số: PM2.5, PM10, SO2, NO2, CO, O3, nhiệt độ, độ ẩm, áp suất, hướng gió, tốc độ gió

---

## 🔄 Quy Trình Thực Hiện (Pipeline)

```
01_Preprocessing_EDA → 02_Feature_Preparation → 03_Classification → 04_Regression → 05_ARIMA_Forecasting
```

---

## 📊 Chi Tiết Từng Notebook

### 1️⃣ Notebook 01: Preprocessing & EDA
**File**: `preprocessing_and_eda_run.ipynb`

#### 📥 Input
- **File ZIP**: `data/raw/PRSA2017_Data_20130301-20170228.zip`
  - Chứa 12 file CSV (1 file/trạm)
  - Khoảng thời gian: 01/03/2013 → 28/02/2017 (4 năm)
  - Tần suất: Đo theo giờ (hourly)
- **Các trạm quan trắc**: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong

#### Mục tiêu
- Tải và làm sạch dữ liệu chất lượng không khí Beijing
- Tạo nhãn phân lớp AQI (Air Quality Index) dựa trên PM2.5
- Tạo các đặc trưng thời gian và lag features
- Khám phá dữ liệu ban đầu (EDA)

#### Các bước thực hiện

**1. Load dữ liệu**
- Có 2 tùy chọn:
  - `USE_UCIMLREPO=True`: Tải trực tiếp từ UCI ML Repository (cần internet)
  - `USE_UCIMLREPO=False`: Đọc từ file ZIP local
- Gộp dữ liệu từ 12 trạm thành 1 DataFrame
- **Raw shape**: ~420,768 dòng × 18 cột

**2. Làm sạch dữ liệu** (`clean_air_quality_df`)
- Tạo cột `datetime` từ các cột year, month, day, hour
- Xử lý missing values (giữ nguyên, xử lý sau)
- Chuẩn hóa tên cột và kiểu dữ liệu
- Sắp xếp theo station và datetime

**3. Tạo nhãn AQI** (`add_pm25_24h_and_label`)
- Tính **rolling mean 24h** của PM2.5 → `pm25_24h`
  - 23 giờ đầu của mỗi trạm sẽ có pm25_24h = NaN
- Phân loại thành **6 mức AQI**:
  - **Good**: PM2.5 < 12 µg/m³
  - **Moderate**: 12 ≤ PM2.5 < 35.5
  - **Unhealthy_for_Sensitive_Groups**: 35.5 ≤ PM2.5 < 55.5
  - **Unhealthy**: 55.5 ≤ PM2.5 < 150.5
  - **Very_Unhealthy**: 150.5 ≤ PM2.5 < 250.5
  - **Hazardous**: PM2.5 ≥ 250.5

**4. Feature Engineering**
- **Time Features** (`add_time_features`):
  - Circular encoding: `hour_sin`, `hour_cos` (sin/cos của giờ)
  - `dow` (day of week: 0=Monday, 6=Sunday)
  - `is_weekend` (0/1)
- **Lag Features** (`add_lag_features`):
  - Tạo lag 1h, 3h, 24h cho: PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
  - Tổng: 10 biến × 3 lags = 30 features mới
  - Dòng đầu tiên sẽ thiếu lag features

**5. EDA (Exploratory Data Analysis)**
- Kiểm tra tỷ lệ missing data
- Phân bố các lớp AQI (biểu đồ bar chart)
- **Phát hiện**: Dataset **imbalanced** nghiêm trọng
  - Unhealthy: ~148,000 dòng (36%)
  - Good: ~14,000 dòng (3%)

#### 📤 Output
**File**: `data/processed/cleaned.parquet`

**Kích thước**: 420,768 dòng × 55 cột

**Cấu trúc file:**

```
Metadata (3 cột):
├─ No              : int - Số thứ tự
├─ datetime        : datetime64 - Timestamp (YYYY-MM-DD HH:00:00)
└─ station         : object - Tên trạm

Dữ liệu gốc (14 cột):
├─ year, month, day, hour : int - Thông tin thời gian
├─ PM2.5           : float - Bụi mịn PM2.5 (µg/m³) ⚠️ LEAKAGE
├─ PM10            : float - Bụi PM10 (µg/m³)
├─ SO2             : float - Sulfur dioxide (µg/m³)
├─ NO2             : float - Nitrogen dioxide (µg/m³)
├─ CO              : float - Carbon monoxide (µg/m³)
├─ O3              : float - Ozone (µg/m³)
├─ TEMP            : float - Nhiệt độ (°C)
├─ PRES            : float - Áp suất (hPa)
├─ DEWP            : float - Điểm sương (°C)
├─ RAIN            : float - Lượng mưa (mm)
├─ wd              : object - Hướng gió (categorical)
└─ WSPM            : float - Tốc độ gió (m/s)

Target Variables (2 cột):
├─ pm25_24h        : float - Rolling mean 24h của PM2.5 ⚠️ LEAKAGE
└─ aqi_class       : object - Nhãn AQI (6 classes) 🎯 TARGET

Time Features (4 cột):
├─ hour_sin        : float - sin(2π * hour/24)
├─ hour_cos        : float - cos(2π * hour/24)
├─ dow             : int - Day of week (0-6)
└─ is_weekend      : int - Cuối tuần (0/1)

Lag Features (30 cột):
├─ PM10_lag1, PM10_lag3, PM10_lag24
├─ SO2_lag1, SO2_lag3, SO2_lag24
├─ NO2_lag1, NO2_lag3, NO2_lag24
├─ CO_lag1, CO_lag3, CO_lag24
├─ O3_lag1, O3_lag3, O3_lag24
├─ TEMP_lag1, TEMP_lag3, TEMP_lag24
├─ PRES_lag1, PRES_lag3, PRES_lag24
├─ DEWP_lag1, DEWP_lag3, DEWP_lag24
├─ RAIN_lag1, RAIN_lag3, RAIN_lag24
└─ WSPM_lag1, WSPM_lag3, WSPM_lag24
```

**Missing Values:**
- 23 giờ đầu của mỗi trạm: `pm25_24h` và `aqi_class` = NaN
- Lag 24h features: Thiếu ở 24 dòng đầu mỗi trạm
- Các biến môi trường: Có missing do lỗi cảm biến

**Sample Data:**
```
datetime             station       PM2.5  pm25_24h  aqi_class  hour_sin  dow  PM10_lag1
2013-03-01 17:00:00  Aotizhongxin  10.0   5.0      Good       0.707     4    23.0
2013-03-01 18:00:00  Aotizhongxin  11.0   5.3      Good       0.866     4    20.0
```

---

### 2️⃣ Notebook 02: Feature Preparation
**File**: `feature_preparation_run.ipynb`

#### 📥 Input
- **File**: `data/processed/cleaned.parquet`
- **Shape**: 420,768 dòng × 55 cột
- **Nguồn**: Output của Notebook 01

#### Mục tiêu
- Đọc dữ liệu đã làm sạch
- Kiểm tra và xử lý data leakage
- Lọc dữ liệu (loại bỏ dòng không có target)
- Chuẩn bị dataset sạch cho classification modeling

#### Các bước thực hiện

**1. Load dữ liệu**
- Đọc file `cleaned.parquet` từ Notebook 01
- Verify shape và columns

**2. Kiểm tra Data Leakage**
- Xác định các cột có nguy cơ leakage:
  - `PM2.5`: Giá trị gốc (target regression, không dùng cho classification)
  - `pm25_24h`: Giá trị rolling mean 24h (có chứa thông tin từ target)
- ⚠️ **Các cột này VẪN GIỮ trong dataset**, nhưng sẽ loại bỏ khi training model
- **Lý do giữ**: Cần cho visualization và debugging

**3. Lọc dữ liệu** (nếu `DROP_ROWS_WITHOUT_TARGET=True`)
- Loại bỏ các dòng có `aqi_class = NaN`
- **Số dòng loại bỏ**: ~7,833 dòng (1.86%)
- **Lý do loại bỏ**:
  - 23 giờ đầu tiên của mỗi trạm (không đủ 24h để tính pm25_24h)
  - Các dòng PM2.5 thiếu quá nhiều trong cửa sổ 24h
- **Dòng còn lại**: 412,935 dòng (98.14%)

**4. Xác định Feature Set**
- **Features được dùng cho modeling**: 51 cột
- **Loại trừ**: `PM2.5`, `pm25_24h`, `aqi_class`, `datetime` (4 cột)
- **Bao gồm**:
  - Pollution indicators (5): PM10, SO2, NO2, CO, O3
  - Weather features (7): TEMP, PRES, DEWP, RAIN, wd, WSPM
  - Time features (4): hour_sin, hour_cos, dow, is_weekend
  - Raw time (4): year, month, day, hour
  - Lag features (30): Lag 1h, 3h, 24h của 10 biến
  - Metadata (1): station (có thể encode)

**5. Lưu Dataset**
- Lưu toàn bộ 55 cột (không loại bỏ leakage columns)
- Format: Parquet (nén hiệu quả, giữ nguyên data types)

#### 📤 Output
**File**: `data/processed/dataset_for_clf.parquet`

**Kích thước**: 412,935 dòng × 55 cột

**Cấu trúc file:**
```
Giống Notebook 01, nhưng:
✅ Đã loại bỏ dòng có aqi_class = NaN
✅ Tất cả dòng còn lại đều có target hợp lệ
✅ Sẵn sàng cho train/test split

Phân bố aqi_class (sau khi lọc):
├─ Unhealthy                        : 148,558 (36.0%)
├─ Moderate                          : 109,549 (26.5%)
├─ Unhealthy_for_Sensitive_Groups   : 64,731 (15.7%)
├─ Very_Unhealthy                   : 56,242 (13.6%)
├─ Hazardous                        : 19,931 (4.8%)
└─ Good                             : 13,924 (3.4%)
```

**Features cho modeling (51 cột):**
```python
drop_cols = {'PM2.5', 'pm25_24h', 'aqi_class', 'datetime'}
feature_cols = [c for c in df.columns if c not in drop_cols]

# Kết quả:
['No', 'year', 'month', 'day', 'hour',
 'PM10', 'SO2', 'NO2', 'CO', 'O3',
 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM',
 'station', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
 'PM10_lag1', 'SO2_lag1', ..., 'WSPM_lag24']  # 51 features
```

**Sample Data:**
```
datet📤 Output

**1. File**: `data/processed/metrics.json`

**Cấu trúc JSON:**
```json
{
  "accuracy": 0.75,
  "f1_macro": 0.68,
  "confusion_matrix": [
    [120, 15, 5, 0, 0, 0],
    [20, 180, 25, 3, 0, 0],
    ...
  ],
  "labels": ["Good", "Moderate", "Unhealthy_for_Sensitive_Groups", 
             "Unhealthy", "Very_Unhealthy", "Hazardous"],
  "report": {
    "Good": {
      "precision": 0.82,
      "recall": 0.71,
      "f1-score": 0.76,
      "support": 2850
    },
    "Moderate": { ... },
    ...,
    "accuracy": 0.75,
    "macro avg": {
      "precision": 0.70,
      "recall": 0.67,
      "f1-score": 0.68
    },
    "weighted avg": {
      "precision": 0.76,
      "recall": 0.75,
      "f1-score": 0.75
    }
  }
}
```

**2. File**: `data/processed/predictions_sample.csv`

**Kích thước**: 5,000 dòng × ~10 cột

**Cấu trúc CSV:**
```
datetime,station,y_true,y_pred,y_pred_proba_Good,y_pred_proba_Moderate,...
2017-01-01 00:00:00,Aotizhongxin,Unhealthy,Unhealthy,0.02,0.15,...
2017-01-01 01:00:00,Aotizhongxin,Unhealthy,Very_Unhealthy,0.01,0.10,...
```

**Các cột:**
- `datetime`: Timestamp
- `station`: Tên trạm
- `y_true`: Nhãn thực tế (string)
- `y_pred`: Nhãn dự đoán (string)
- `y_pred_proba_*`: Xác suất cho từng class (6 cột)

#### Insights
- Mô hình classification có thể giúp:
  - Cảnh báo mức độ ô nhiễm theo thời gian thực
  - Phân loại chất lượng không khí cho người dân
  - Ra quyết định về hoạt động ngoài trời
  - Planning cho urban development

**Thách thức:**
- **Imbalanced classes**: Good (3%) vs Unhealthy (36%)
- **Confusion giữa các lớp kế cận**: Moderate ↔ Unhealthy_for_Sensitive_Groups
- **Hazardous ít data**: Khó học pattern, recall thấp
- **C📤 Output

**1. File**: `data/processed/dataset_for_regression.parquet`

**Kích thước**: 420,768 dòng × 57 cột

**Cấu trúc:**
```
Giống dataset_for_clf.parquet, nhưng:
- Thêm cột 'y' (float): Target = PM2.5 tại t+1h (horizon=1)
- Không có cột 'aqi_class'
- Giữ cột 'PM2.5' (giá trị hiện tại tại t)

Cột mới:
└─ y : float - PM2.5 tại t+1h (shifted -1) 🎯 TARGET

Missing values:
- Dòng cuối cùng của mỗi trạm: y = NaN (không có giá trị tương lai)
```

**2. File**: `data/processed/regressor.joblib`

**Loại file**: Joblib serialized file (sklearn model)

**Nội dung**: Trained regression model (Random Forest / XGBoost)

**Cách load:**
```python
import joblib
model = joblib.load('data/processed/regressor.joblib')
y_pred = model.predict(X_test)
```

**3. File**: `data/processed/regression_metrics.json`

**Cấu trúc JSON:**
```json
{
  "rmse": 28.45,
  "mae": 18.32,
  "r2": 0.67,
  "train_samples": 380000,
  "test_samples": 40000,
  "features_used": 50,
  "model_type": "RandomForestRegressor",
  "cutoff_date": "2017-01-01"
}
```

**4. File**: `data/processed/regression_predictions_sample.csv`

**Kích thước**: 5,000 dòng × 4 cột

**Cấu trúc CSV:**
```
datetime,station,y_true,y_pred
2017-01-01 00:00:00,Aotizhongxin,85.3,78.5
2017-01-01 01:00:00,Aotizhongxin,82.1,80.2
2017-01-01 02:00:00,Aotizhongxin,79.8,81.5
```

**Các cột:**
- `datetime`: Timestamp của prediction
- `station`: Tên trạm
- `y_true`: Giá trị PM2.5 thực tế tại t+1h
- `y_pred`: Giá trị PM2.5 dự đoán tại t+1h

#### Các bước thực hiện

**1. Load Dataset**
- Đọc `dataset_for_clf.parquet`
- Shape: ~400,000+ dòng

**2. Time-Based Split**
- **Cutoff date**: `2017-01-01`
- **Train**: Dữ liệu trước 2017-01-01
- **Test**: Dữ liệu từ 2017-01-01 trở đi
- **Lý do**: Với time series, phải split theo thời gian để tránh data leakage

**3. Feature Selection**
- Loại bỏ các cột:
  - `datetime`: không phải feature số
  - `station`: categorical, có thể one-hot encode nếu cần
  - `PM2.5`, `pm25_24h`: data leakage
  - `aqi_class`: target variable

**4. Training Classifier**
- Mô hình: Có thể là Random Forest, XGBoost, hoặc ensemble
- Sử dụng hàm `train_classifier` từ `src.classification_library`
- Pipeline bao gồm:
  - Preprocessing (scaling, encoding)
  - Model training
  - Prediction trên test set

**5. Evaluation Metrics**
- **Accuracy**: Độ chính xác tổng thể
- **F1-macro**: Trung bình F1 của tất cả các lớp (không weight theo số lượng)
- **Confusion Matrix**: Ma trận nhầm lẫn giữa các lớp
- **Classification Report**: Precision, Recall, F1 cho từng lớp

**6. Visualization**
- Plot confusion matrix với heatmap
- Phân tích lỗi: lớp nào bị nhầm nhiều nhất

#### Output
- **Metrics**: `data/processed/metrics.json`
  - Accuracy, F1-macro
  - Confusion matrix
  - Classification report chi tiết
- **Predictions**: `data/processed/predictions_sample.csv`
  - Sample 5000 dòng đầu
  - Gồm: y_true, y_pred, features

#### Insights
- Mô hình classification có thể giúp:
  - Cảnh báo mức độ ô nhiễm
  - Phân loại chất lượng không khí real-time
  - Ra quyết định về hoạt động ngoài trời
- Có thể kém hiệu quả với các lớp có ít sample (Hazardous)

---

### 4️⃣ Notebook 04: Regression Modelling
**File**: `regression_modelling_run.ipynb`

#### Mục tiêu
- Chuyển bài toán time series thành supervised regression
- Dự đoán giá trị PM2.5 tại thời điểm t+h từ features tại thời điểm t
- So sánh với phương pháp ARIMA (sẽ làm ở notebook 05)

#### Điểm khác biệt với Classification
- **Target**: Giá trị liên tục PM2.5 (không phải lớp AQI)
- **Horizon**: Dự đoán trước h giờ (mặc định h=1)
- **Approach**: Feature-based regression thay vì time-series-based

#### Các bước thực hiện

**1. Tạo Regression Dataset** (`run_prepare_regression_dataset`)
- Load dữ liệu gốc
- Tạo target: `y = PM2.5(t+h)` với horizon=1
  - Shift PM2.5 xuống 1 bước (tương lai 1h)
  - Dòng cuối cùng sẽ không có target
- Tạo lag features: PM2.5(t-1), PM2.5(t-3), PM2.5(t-24)
- Tạo time features
- Lưu: `data/processed/dataset_for_regression.parquet`

**2. EDA cho Regression Dataset**
- **Missing values**: Kiểm tra tỷ lệ missing ở lag features
  - Thường thiếu ở đầu chuỗi (không đủ history)
- **Distribution**: Phân bố PM2.5
  - Có thể skewed → cân nhắc log transform
  - Có outliers → cân nhắc clip values
- **Seasonality**: Kiểm tra pattern theo giờ/ngày

**3. Time-Based Split**
- **Cutoff**: `2017-01-01`
- Train trước cutoff, test sau cutoff
- **Critical**: Với time series regression, vẫn phải split theo thời gian

**4. Train Regressor** (`run_train_regression`)
- Mod📤 Output

**1. File**: `data/processed/arima_pm25_predictions.csv`

**Kích thước**: Số dòng = test set length (ví dụ: ~10,000 giờ)

**Cấu trúc CSV:**
```
datetime,y_true,y_pred,lower,upper
2017-01-01 00:00:00,85.3,78.5,65.2,91.8
2017-01-01 01:00:00,82.1,80.2,67.1,93.3
2017-01-01 02:00:00,79.8,81.5,68.5,94.5
...
```

**Các cột:**
- `datetime`: Timestamp
- `y_true`: Giá trị PM2.5 thực tế (float)
- `y_pred`: Forecast từ ARIMA (float)
- `lower`: Lower bound của 95% confidence interval (float)
- `upper`: Upper bound của 95% confidence interval (float)

**2. File**: `arima_pm25_model.pkl`

**Loại file**: Pickle serialized file (statsmodels ARIMAResults)

**Cách load:**
```python
import pickle
with open('data/processed/arima_pm25_model.pkl', 'rb') as f:
    model = pickle.load(f)
forecast = model.forecast(steps=24)  # Dự báo 24h tiếp theo
```

**Nội dung**: Trained ARIMA model với fitted parameters

**3. File**: `data/processed/arima_pm25_summary.json`

**Cấu trúc JSON:**
```json
{
  "station": "Aotizhongxin",
  "value_col": "PM2.5",
  "cutoff": "2017-01-01",
  "best_order": [2, 1, 1],
  "ic": "aic",
  "best_score": 125430.56,
  "rmse": 32.18,
  "mae": 22.45,
  "diagnostics": {
    "n_obs": 35064,
    "missing_pct": 0.0,
    "mean": 89.45,
    "std": 76.32,
    "min": 1.0,
    "max": 999.0,
    "adf_statistic": -12.45,
    "adf_pvalue": 0.0,
    "adf_is_stationary": true,
    "kpss_statistic": 0.23,
    "kpss_pvalue": 0.1,
    "kpss_is_stationary": true
  }
}
```

**Chi tiết các field:**
- `station`: Tên trạm phân tích
- `value_col`: Biến được forecast (PM2.5)
- `cutoff`: Ngày chia train/test
- `best_order`: [p, d, q] - ARIMA order tối ưu
- `ic`: Information criterion dùng (aic/bic)
- `best_score`: Giá trị AIC/BIC thấp nhất
- `rmse`, `mae`: Lỗi trên test set
- `diagnostics`: Thống kê chuỗi thời gian
  - Stationarity tests (ADF, KPSS)
  - Summary statistics
  - Missing data percentage
- **Visualization**:
  - Plot actual vs predicted trong cửa sổ thời gian
  - Giúp nhìn thấy pattern và errors

#### Output
- **Model**: `data/processed/regressor.joblib`
- **Metrics**: `data/processed/regression_metrics.json`
  - RMSE, MAE, R²
- **Predictions**: `data/pr - Chi Tiết

```
data/
├── raw/
│   └── PRSA2017_Data_20130301-20170228.zip     # Input data (12 CSV files)
│       ├── PRSA_Data_Aotizhongxin_20130301-20170228.csv
│       ├── PRSA_Data_Changping_20130301-20170228.csv
│       └── ... (10 trạm khác)
│
└── processed/
    │
    ├─── 📊 NOTEBOOK 01 OUTPUT ───
    ├── cleaned.parquet                         # 420,768 × 55 - Dữ liệu đầy đủ
    │   │   Columns: 55 (metadata + raw + target + time + lag features)
    │   │   Bao gồm: PM2.5, pm25_24h, aqi_class, và TẤT CẢ features
    │   │   Missing: pm25_24h/aqi_class NaN ở 23h đầu mỗi trạm
    │   └── Format: Parquet (compressed, preserves dtypes)
    │
    ├─── 📊 NOTEBOOK 02 OUTPUT ───
    ├── dataset_for_clf.parquet                 # 412,935 × 55 - Sạch cho classification
    │   │   Columns: Giống cleaned.parquet
    │   │   Khác biệt: Đã loại dòng có aqi_class = NaN
    │   │   Ready for: Train/test split và modeling
    │   └── Features: 51 (loại trừ PM2.5, pm25_24h, aqi_class, datetime)
    │
    ├─── 📊 NOTEBOOK 03 OUTPUT ───
    ├── metrics.json                            # ~2 KB - Classification metrics
    │   │   Chứa: accuracy, f1_macro, confusion_matrix, classification_report
    │   └── Format: JSON (human-readable)
    │
    ├── predictions_sample.csv                  # 5,000 × ~10 - Sample predictions
    │   │   Columns: datetime, station, y_true, y_pred, probabilities (6 classes)
    │   └── Format: CSV
    │
    ├─── 📊 NOTEBOOK 04 OUTPUT ───
    ├── dataset_for_regression.parquet          # 420,768 × 57 - Dataset cho regression
    │   │   Columns: 56 (giống clf) + 1 (cột 'y' = PM2.5 tại t+1h)
    │   │   Target: y (shifted PM2.5)
    │   └── Missing: Dòng cuối mỗi trạm (không có future value)
    │
    ├── regressor.joblib                        # ~50-200 MB - Trained model
    │   │   Type: Random Forest / XGBoost Regressor
    │   │   Can load: joblib.load()
    │   └── Use: model.predict(X_new)
    │
    ├── regression_metrics.json                 # ~1 KB - Regression metrics
    │   │   Chứa: rmse, mae, r2, train/test counts
    │   └── Format: JSON
    │
    ├── regression_predictions_sample.csv       # 5,000 × 4 - Sample predictions
    │   │   Columns: datetime, station, y_true, y_pred
    │   └── Format: CSV
    │
    ├─── 📊 NOTEBOOK 05 OUTPUT ───
    ├── arima_pm25_predictions.csv              # ~10,000 × 5 - ARIMA forecasts
    │   │   Columns: datetime, y_true, y_pred, lower, upper (CI bounds)
    │   │   Chỉ cho 1 trạm: Aotizhongxin
    │   └── Format: CSV
    │
    ├── arima_pm25_model.pkl                    # ~1 MB - ARIMA model object
    │   │   Type: statsmodels ARIMAResults
    │   │   Can load: pickle.load()
    │   └── Use: model.forecast(steps=n)
    │
    └── arima_pm25_summary.json                 # ~2 KB - ARIMA metadata
        │   Chứa: best_order, AIC, RMSE, MAE, stationarity tests
        └── Format: JSON
```

### 📊 Tổng Kích Thước Files

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `cleaned.parquet` | ~80 MB | Full dataset với tất cả features |
| `dataset_for_clf.parquet` | ~75 MB | Đã lọc, sẵn sàng train |
| `dataset_for_regression.parquet` | ~85 MB | Thêm cột target 'y' |
| `regressor.joblib` | ~50-200 MB | Tùy model (RF/XGBoost) |
| `arima_pm25_model.pkl` | ~1 MB | ARIMA model nhẹ |
| `*.json` | <5 KB mỗi file | Metadata và metrics |
| `*.csv` | <1 MB mỗi file | Sample predictions |
| **TỔNG** | ~300-400 MB | Toàn bộ output |
#### Mục tiêu
- Dự báo chuỗi thời gian PM2.5 bằng mô hình ARIMA
- Hiểu về trend, seasonality, stationarity
- Grid search tham số (p,d,q) tối ưu
- So sánh với regression approach

#### Background: ARIMA
**ARIMA(p,d,q)** = AutoRegressive Integrated Moving Average
- **AR(p)**: AutoRegressive - dựa vào p giá trị quá khứ
- **I(d)**: Integrated - differencing d lần để đạt stationarity
- **MA(q)**: Moving Average - dựa vào q lỗi dự đoán quá khứ

#### Các bước thực hiện

**1. Prepare Time Series**
- Chọn **1 trạm** để phân tích: `Aotizhongxin`
- Chỉ lấy **1 biến**: `PM2.5` (univariate)
- Tạo chuỗi hourly với `make_hourly_station_series`:
  - Frequency: 'H' (hourly)
  - Fill method: 'interpolate_time' (nội suy tuyến tính)
- Length: ~35,000 giờ (4 năm data)

**2. Time Series Diagnostics** (`describe_time_series`)

**a) Stationarity Tests**
- **ADF (Augmented Dickey-Fuller)**:
  - H0: Series có unit root (non-stationary)
  - Nếu p-value < 0.05 → reject H0 → stationary
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**:
  - H0: Series is stationary
  - Nếu p-value < 0.05 → reject H0 → non-stationary
- **Kết luận**: Cần xem cả 2 test cùng nhau

**b) Visual Checks**
- **Plot raw series**: Nhìn trend, jumps, gaps
- **Rolling statistics**: 
  - Rolling mean (7 days)
  - Rolling std (7 days)
  - Nếu mean/std thay đổi theo thời gian → non-stationary

**c) Seasonality**
- **Hourly pattern**: Average by hour-of-day
  - Peak vào giờ nào? (thường buổi sáng/chiều cao điểm)
- **Weekly pattern**: Average by day-of-week
  - Tuần có khác cuối tuần không?

**3. ACF/PACF Analysis**
- **ACF (AutoCorrelation Function)**: 
  - Correlation với các lag
  - Giảm dần → suggest MA order
- **PACF (Partial AutoCorrelation Function)**:
  - Correlation trực tiếp
  - Cut-off point → suggest AR order
- Plot 72 lags (3 days)

**4. Grid Search ARIMA** (`grid_search_arima_order`)
- **Parameter ranges**:
  - p_max = 3 (AR order)
  - d_max = 2 (differencing)
  - q_max = 3 (MA order)
- **Selection criterion**: AIC (Akaike Information Criterion)
  - Hoặc BIC (Bayesian IC)
  - Càng thấp càng tốt
- **Process**: Thử tất cả combinations (p,d,q)
- **Output**: Best order, e.g., ARIMA(2,1,1)

**5. Fit & Forecast** (`fit_arima_and_forecast`)
- Fit ARIMA với best order trên train set
- Forecast n bước (n = len(test))
- Output:
  - Point forecast
  - 95% Confidence Interval

**6. Evaluation**
- **Metrics**:
  - **RMSE**: Root Mean Squared Error
  - **MAE**: Mean Absolute Error
- **Visualization**:
  - Plot actual vs forecast
  - Confidence interval bands
  - Chỉ plot 14 ngày đầu để dễ nhìn

#### Output
- **Predictions**: `data/processed/arima_pm25_predictions.csv`
  - datetime, y_true, y_pred, lower, upper
- **Model**: `arima_pm25_model.pkl`
- **Summary**: `data/processed/arima_pm25_summary.json`
  - Best order: (p,d,q)
  - AIC score
  - RMSE, MAE
  - Stationarity diagnostics

#### Key Insights

**Ưu điểm của ARIMA**:
- Không cần external features
- Tự động học temporal structure
- Confidence intervals built-in
- Interpretable parameters

**Nhược điểm**:
- Chỉ dự báo 1 biến (univariate)
- Không sử dụng được weather/pollution factors khác
- Giả định linear relationships
- Performance giảm khi forecast xa

**Khi nào dùng ARIMA vs Regression?**
- **ARIMA**: Khi chỉ có history của biến đó, cần forecast dài hạn
- **Regression**: Khi có nhiều predictors, cần explain relationships
- **Hybrid**: Có thể combine cả 2 (ARIMAX, VAR)

---

## 📁 Cấu Trúc Output Files

```
data/processed/
├── cleaned.parquet                      # [NB01] Dữ liệu sau preprocessing
├── dataset_for_clf.parquet              # [NB02] Dataset cho classification
├── dataset_for_regression.parquet       # [NB04] Dataset cho regression
├── metrics.json                         # [NB03] Classification metrics
├── predictions_sample.csv               # [NB03] Classification predictions
├── regressor.joblib                     # [NB04] Trained regression model
├── regression_metrics.json              # [NB04] Regression metrics
├── regression_predictions_sample.csv    # [NB04] Regression predictions
├── arima_pm25_predictions.csv           # [NB05] ARIMA forecast results
├── arima_pm25_model.pkl                 # [NB05] Trained ARIMA model
└── arima_pm25_summary.json              # [NB05] ARIMA summary & diagnostics
```

---

## 🔑 Key Takeaways

### 1. Data Leakage Prevention
- **Vấn đề**: PM2.5 và pm25_24h chứa thông tin tương lai
- **Giải pháp**: Loại khỏi features, chỉ giữ làm target
- **Lesson**: Luôn kiểm tra temporal relationships trong features

### 2. Time-Based Split
- **Không được** dùng random split với time series
- **Phải** split theo thời gian: train trước, test sau
- **Lý do**: Tránh training trên future information

### 3. Feature Engineering cho Time Series
- **Lag features**: Giá trị quá khứ (t-1, t-3, t-24)
- **Time features**: hour, day, month, season, is_weekend
- **Domain features**: Weather, pollution indicators
- **Trade-off**: Nhiều features → complex model, nhưng better performance

### 4. Classification vs Regression
- **Classification**: Phân loại mức độ (Good/Moderate/Unhealthy...)
  - Dễ interpret cho decision making
  - Mất thông tin giá trị chính xác
- **Regression**: Dự đoán giá trị liên tục
  - Giữ được thông tin chi tiết
  - Khó interpret cho non-technical users

### 5. Feature-Based vs Time-Series Models
- **Feature-Based (RF/XGBoost)**:
  - Sử dụng nhiều biến
  - Capture non-linear relationships
  - Cần feature engineering
  - Short-term forecast tốt
  
- **Time-Series (ARIMA)**:
  - Univariate, chỉ dùng history
  - Linear assumptions
  - Auto-learns temporal patterns
  - Có confidence intervals
  - Medium-term forecast

### 6. Model Selection Strategy
```
Có nhiều external predictors (weather)?
├─ YES → Feature-based Regression/Classification
│         (Random Forest, XGBoost, Neural Network)
└─ NO  → Time Series Models
          ├─ Stationary? → ARIMA
          ├─ Complex seasonality? → SARIMA, Prophet
          └─ Multiple variables? → VAR, VARIMA
```

---

## 🛠️ Technical Stack

### Python Libraries
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Time Series**: `statsmodels`
- **I/O**: `joblib` (model persistence)

### Custom Modules (src/)
- `classification_library.py`: Data loading, preprocessing, classification utils
- `regression_library.py`: Regression dataset prep, training, evaluation
- `timeseries_library.py`: ARIMA utils, stationarity tests, forecasting

---

## 🚀 Cách Chạy Pipeline

### Option 1: Chạy từng notebook thủ công
```bash
jupyter notebook notebooks/runs/preprocessing_and_eda_run.ipynb
# ... tiếp tục với các notebooks khác
```

### Option 2: Chạy tự động bằng Papermill
```bash
python run_papermill.py
```

Script `run_papermill.py` sẽ chạy lần lượt tất cả 5 notebooks với parameters đã định nghĩa.

---

## 📈 Kết Quả & Metrics

### Classification (AQI Level Prediction)
- **Accuracy**: ~70-80% (tùy model)
- **F1-macro**: ~0.65-0.75
- **Challenge**: Imbalanced classes (Hazardous rất ít)

### Regression (PM2.5 Value Prediction)
- **RMSE**: ~20-30 µg/m³
- **MAE**: ~15-20 µg/m³
- **R²**: ~0.6-0.7

### ARIMA Forecasting
- **Best Order**: Thường ARIMA(1-3, 1, 1-2)
- **RMSE**: ~25-35 µg/m³ (hourly forecast)
- **MAE**: ~18-25 µg/m³

---

## 🎯 Ứng Dụng Thực Tế

1. **Public Health Alerts**
   - Cảnh báo chất lượng không khí real-time
   - Khuyến nghị hoạt động ngoài trời

2. **Urban Planning**
   - Quyết định về giao thông, công nghiệp
   - Đánh giá hiệu quả chính sách môi trường

3. **Personal Protection**
   - App dự báo AQI cho người dân
   - Scheduling outdoor activities

4. **Research**
   - Hiểu factors ảnh hưởng đến ô nhiễm
   - Modeling climate impact

---

## 📚 Tài Liệu Tham Khảo

- Dataset: [Beijing Multi-Site Air Quality - UCI ML Repository](https://archive.ics.uci.edu/dataset/501/)
- AQI Standards: [EPA Air Quality Index](https://www.airnow.gov/aqi/)
- ARIMA Tutorial: [statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

---

## ✅ Checklist Hoàn Thành

- [x] Notebook 01: Preprocessing & EDA
- [x] Notebook 02: Feature Preparation
- [x] Notebook 03: Classification Modelling
- [x] Notebook 04: Regression Modelling
- [x] Notebook 05: ARIMA Forecasting
- [x] Tạo tài liệu tổng hợp

---

**Người thực hiện**: GitHub Copilot  
**Ngày hoàn thành**: 18/01/2026  
**Version**: 1.0
