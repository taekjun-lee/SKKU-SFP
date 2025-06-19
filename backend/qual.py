# Module Import
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import json
from IPython.display import display
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GRU, LSTM, Bidirectional,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# 데이터 분할 학습(60%), 검증(40%)
total_rows = len(data)
split_idx = int(total_rows * 0.6)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# 트렌드 예측 window 생성
def create_sensor_trend_windows(data, sensor_cols, window_size=60, step=1):
    sensor_data = data[sensor_cols].to_numpy(dtype=np.float32)
    num_windows = (len(sensor_data) - window_size) // step

    X = np.empty((num_windows, window_size, len(sensor_cols)), dtype=np.float32)
    y = np.empty((num_windows, len(sensor_cols)), dtype=np.float32)

    for i in range(num_windows):
        idx = i * step
        X[i] = sensor_data[idx:idx + window_size]
        y[i] = sensor_data[idx + window_size]
    return X, y

X_train, y_train = create_sensor_trend_windows(train_data, sensor_cols)
X_test, y_test = create_sensor_trend_windows(test_data, sensor_cols)


# 주요 모델 전체 데이터로 학습 (GRU, BiLSTM, LSTM)
# 실제 웹 배포 시 모델 학습/평가 등의 코드는 별도 python에서 수행 후 선정된 모델의 코드만 남기면 될듯
# 프로젝트 발표를 위해 비교군 모델에 대한 데이터도 일단 수행
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# GRU
gru_trend_model = Sequential([
    GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
gru_trend_model.compile(
    optimizer='adam',
    loss='mse'
)
gru_trend_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)
gru_trend_model.save("data/gru_trend_model.h5")

# BiLSTM
bilstm_trend_model = Sequential([
    Bidirectional(LSTM(128), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
bilstm_trend_model.compile(
    optimizer='adam',
    loss='mse'
)
bilstm_trend_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)
bilstm_trend_model.save("data/bilstm_trend_model.h5")

# LSTM
lstm_trend_model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
lstm_trend_model.compile(
    optimizer='adam',
    loss='mse'
)
lstm_trend_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)
lstm_trend_model.save("data/lstm_trend_model.h5")

# GRU, BiLSTM, LSTM 학습 평가
models_info = {
    "GRU": "data/gru_trend_model.h5",
    "LSTM": "data/lstm_trend_model.h5",
    "BiLSTM": "data/bilstm_trend_model.h5"
}

all_metrics = {}

for model_name, file_path in models_info.items():
    model = load_model(file_path, compile=False)

    X_init = train_data[sensor_cols].values[-60:].reshape(1, 60, len(sensor_cols))
    first_y_pred = model.predict(X_init)
    y_pred_rest = model.predict(X_test)
    y_pred_full = np.vstack([first_y_pred, y_pred_rest])

    metrics = []
    for i, sensor in enumerate(sensor_cols):
        pred = y_pred_full[:, i]
        actual = y_test[:, i]

        mse = mean_squared_error(actual, pred[1:])
        mae = mean_absolute_error(actual, pred[1:])
        r2 = r2_score(actual, pred[1:])

        metrics.append({
            "Sensor": sensor,
            "MSE": round(mse, 5),
            "MAE": round(mae, 5),
            "R2": round(r2, 5)
        })

    df = pd.DataFrame(metrics)
    df["Model"] = model_name
    all_metrics[model_name] = df

combined_df = pd.concat(all_metrics.values(), ignore_index=True)

avg_summary = {
    model: {
        "Average_MSE": round(df["MSE"].mean(), 5),
        "Average_MAE": round(df["MAE"].mean(), 5),
        "Average_R2": round(df["R2"].mean(), 5)
    }
    for model, df in all_metrics.items()
}

output_json = {
    "details": combined_df.to_dict(orient="records"),
    "summary": avg_summary
}

with open("data/model_metrics_comparison.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

gru_df = all_metrics["GRU"][["Sensor", "MSE", "MAE", "R2"]].copy()

gru_metric_output = {
    "details": gru_df.to_dict(orient="records")
}

with open("data/gru_metrics.json", "w", encoding="utf-8") as f:
    json.dump(gru_metric_output, f, indent=2, ensure_ascii=False)

window_bank, next_step_bank = [], []
train_arr = train_data[sensor_cols].values
for i in range(len(train_arr) - 60 - 1):
    window_bank.append(train_arr[i:i+60].flatten())
    next_step_bank.append(train_arr[i+60])
window_bank = np.array(window_bank)
next_step_bank = np.array(next_step_bank)
knn_model = NearestNeighbors(n_neighbors=3).fit(window_bank)

X_init = train_arr[-60:]
gru_first_pred = gru_trend_model.predict(X_init.reshape(1, 60, len(sensor_cols)))
gru_test_pred = gru_trend_model.predict(X_test)
gru_pred_full = np.vstack([gru_first_pred, gru_test_pred])

future_steps = 10080
future_preds = np.empty((future_steps, len(sensor_cols)))
last_input = X_test[-1:].copy()

for step in range(future_steps):
    pred = gru_trend_model.predict(last_input, verbose=0)[0]

    query = last_input.reshape(1, -1)
    indices = knn_model.kneighbors(query, return_distance=False)
    similar_nexts = next_step_bank[indices[0]]

    corrected = (pred + np.mean(similar_nexts, axis=0)) / 2
    future_preds[step] = corrected

    last_input = np.concatenate([last_input[:, 1:, :], corrected.reshape(1, 1, -1)], axis=1)

train_values_orig = scaler.inverse_transform(train_data[sensor_cols].values)
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(gru_pred_full)
future_pred_orig  = scaler.inverse_transform(future_preds)

time_gap = test_data.index[1] - test_data.index[0]
start_time = train_data.index[-1] + time_gap
pred_timestamps = [start_time + i * time_gap for i in range(len(y_pred_orig))]
future_timestamps = [pred_timestamps[-1] + (i+1)*time_gap for i in range(future_steps)]

results = []
for i, sensor in enumerate(sensor_cols):
    result = {
        "sensor": sensor,
        "train_timestamps": train_data.index.astype(str).tolist(),
        "train_values": train_values_orig[:, i].tolist(),
        "test_timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in pred_timestamps + future_timestamps],
        "predicted_values": y_pred_orig[:, i].tolist() + future_pred_orig[:, i].tolist(),
        "actual_values": y_test_orig[:, i].tolist() + [None] * future_steps
    }
    results.append(result)

with open("data/gru_forecast.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

