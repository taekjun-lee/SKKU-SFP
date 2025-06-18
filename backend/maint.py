# Module Import
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import json
from IPython.display import display
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GRU, LSTM, Bidirectional,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tcn import TCN

# 데이터 전처리
data = pd.read_csv("../frontend/public/sensor.csv")

all_zero_cols = data.columns[(data == 0).all()].tolist()
all_nan_cols = data.columns[data.isna().all()].tolist()
threshold = 1.2
low_std_cols = data.std(numeric_only=True).loc[lambda x: x < threshold].index.tolist()
useless_columns = list(set(all_zero_cols + all_nan_cols + low_std_cols))
data = data.drop(columns=useless_columns)

status_map = {'NORMAL': 0, 'RECOVERING': 1, 'BROKEN': 2}
data['machine_status_encoded'] = data['machine_status'].map(status_map)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data = data.interpolate(method='time')

sensor_cols = [col for col in data.columns if col.startswith('sensor')]
scaler = StandardScaler()
data[sensor_cols] = scaler.fit_transform(data[sensor_cols])

data = data.drop(columns=['machine_status'])
data = data.sort_index()

# 데이터 분할 학습(60%), 테스트(40%)
total_rows = len(data)
split_idx = int(total_rows * 0.6)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# 상태 예측 window 생성
def create_windows_fast(data, window_size=60, step=1, label_col='machine_status_encoded'):
    sensor_cols = [col for col in data.columns if col.startswith('sensor')]
    sensor_data = data[sensor_cols].to_numpy(dtype=np.float32)
    label_data = data[label_col].to_numpy(dtype=np.int8)

    num_windows = (len(sensor_data) - window_size) // step
    X = np.empty((num_windows, window_size, len(sensor_cols)), dtype=np.float32)
    y = np.empty(num_windows, dtype=np.int8)

    for i in range(num_windows):
        idx = i * step
        X[i] = sensor_data[idx:idx + window_size]
        y[i] = label_data[idx + window_size]

    return X, y

X_train, y_train = create_windows_fast(train_data)
X_test, y_test = create_windows_fast(test_data)

# XGBoost
# 실제 웹 배포 시 모델 학습/평가 등의 코드는 별도 python에서 수행 후 선정된 모델의 코드만 남기면 될듯
# 프로젝트 발표를 위해 비교군 모델에 대한 데이터도 일단 수행
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)

xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
y_pred_xgb=y_pred

xgb_model.save_model("xgb_model.json")

report_xgb = classification_report(
    y_test, y_pred, target_names=['NORMAL', 'RECOVERING', 'BROKEN'], output_dict=True
)

# LightGBM
lgbm_model = lgb.LGBMClassifier(
    n_estimators      = 200,
    max_depth        = -1,
    learning_rate    = 0.05,
    num_leaves       = 64,
    objective        = 'multiclass',
    class_weight     = None,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    n_jobs           = -1,
    random_state     = 42
)

lgbm_model.fit(
    X_train.reshape(X_train.shape[0], -1),
    y_train
)

y_pred = lgbm_model.predict(
    X_test.reshape(X_test.shape[0], -1)
)
y_pred_lgbm=y_pred

lgbm_model.booster_.save_model("lgbm_model.txt")

report_lgbm = classification_report(
    y_test, y_pred, target_names=['NORMAL', 'RECOVERING', 'BROKEN'], output_dict=True
)

# 1D-CNN
y_train_cnn = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_cnn = tf.keras.utils.to_categorical(y_test, num_classes=3)

cnn_model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = cnn_model.fit(
    X_train, y_train_cnn,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

y_pred_proba = cnn_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_cnn, axis=1)
y_pred_cnn=y_pred

cnn_model.save("cnn_model.h5")

report_cnn = classification_report(
    y_true, y_pred, target_names=['NORMAL', 'RECOVERING', 'BROKEN'], output_dict=True
)

# LSTM
y_train_lstm = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_lstm = tf.keras.utils.to_categorical(y_test, num_classes=3)

lstm_model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

lstm_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_train, y_train_lstm,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

y_pred_proba = lstm_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_lstm, axis=1)
y_pred_lstm=y_pred

lstm_model.save("lstm_model.h5")

report_lstm = classification_report(
    y_true, y_pred, target_names=['NORMAL', 'RECOVERING', 'BROKEN'], output_dict=True
)

# 전체 예측 F1-score 비교
f1_scores = pd.DataFrame({
    'XGBoost':    [report_xgb[c]['f1-score'] for c in ['NORMAL', 'RECOVERING', 'BROKEN']],
    'LightGBM':   [report_lgbm[c]['f1-score'] for c in ['NORMAL', 'RECOVERING', 'BROKEN']],
    '1D-CNN':     [report_cnn[c]['f1-score'] for c in ['NORMAL', 'RECOVERING', 'BROKEN']],
    'LSTM':       [report_lstm[c]['f1-score'] for c in ['NORMAL', 'RECOVERING', 'BROKEN']],
}, index=['NORMAL', 'RECOVERING', 'BROKEN'])

f1_scores_dict = {
    "index": f1_scores.index.tolist(),
    "columns": f1_scores.columns.tolist(),
    "data": f1_scores.values.tolist()
}

with open("f1_scores.json", "w") as f:
    json.dump(f1_scores_dict, f, indent=2)

# 전체 예측 결과 차트로 비교
test_timestamps = test_data.index.to_series().reset_index(drop=True)
y_true = test_data['machine_status_encoded'].values

def pad_prediction(y_pred, target_len):
    pad_len = target_len - len(y_pred)
    return np.concatenate([np.full(pad_len, np.nan), y_pred])

y_pred_xgb   = pad_prediction(y_pred_xgb,   len(test_timestamps))
y_pred_lgbm  = pad_prediction(y_pred_lgbm,  len(test_timestamps))
y_pred_cnn   = pad_prediction(y_pred_cnn,   len(test_timestamps))
y_pred_lstm  = pad_prediction(y_pred_lstm,  len(test_timestamps))

result_df = pd.DataFrame({
    "timestamp": test_timestamps.astype(str),
    "actual": y_true.tolist(),
    "XGBoost": y_pred_xgb.tolist(),
    "LightGBM": y_pred_lgbm.tolist(),
    "1D-CNN": y_pred_cnn.tolist(),
    "LSTM": y_pred_lstm.tolist()
})
result_df.to_json("prediction_results.json", orient="records", indent=2)

# RECOVERING 시점 데이터
train_timestamps = train_data.index.to_series().reset_index(drop=True)
train_labels = train_data["machine_status_encoded"].values

train_rec_times = train_timestamps[(train_labels == 1)].astype(str).tolist()
test_rec_times = test_timestamps[(y_pred_xgb == 1)].astype(str).tolist()

all_rec_times = sorted(set(train_rec_times + test_rec_times))

recovering_regions = [{
    "recovering_timestamps": all_rec_times
}]

with open("recovering_regions.json", "w", encoding="utf-8") as f:
    json.dump(recovering_regions, f, indent=2, ensure_ascii=False)