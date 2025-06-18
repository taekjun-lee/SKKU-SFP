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

# 데이터 축소 (10% 샘플링)
# 아래의 약식 평가는 실제 웹 배포 시 불필요 (이미 모델 선정 완료했을테니)
# 프로젝트 발표를 위해 비교군 모델에 대한 데이터도 일단 수행
subset_size = int(len(X_train) * 0.1)
X_train_small = X_train[:subset_size]
y_train_small = y_train[:subset_size]

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

def evaluate_model(name, y_true, y_pred):
    mse_list = [mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(len(sensor_cols))]
    return pd.DataFrame({'Sensor': sensor_cols, f'{name}_MSE': mse_list})

# GRU
gru_model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
gru_model.compile(optimizer='adam', loss='mse')
gru_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
gru_y_pred = gru_model.predict(X_test)
df_gru = evaluate_model('GRU', y_test, gru_y_pred)

# LSTM
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
lstm_y_pred = lstm_model.predict(X_test)
df_lstm = evaluate_model('LSTM', y_test, lstm_y_pred)

# BiLSTM
bilstm_model = Sequential([
    Bidirectional(LSTM(64), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
bilstm_model.compile(optimizer='adam', loss='mse')
bilstm_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
bilstm_y_pred = bilstm_model.predict(X_test)
df_bilstm = evaluate_model('BiLSTM', y_test, bilstm_y_pred)

# TCN
tcn_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    TCN(nb_filters=32, kernel_size=3, dilations=[1, 2, 4], dropout_rate=0.1),
    Dense(len(sensor_cols))
])
tcn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
tcn_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0, callbacks=[early_stop])
tcn_y_pred = tcn_model.predict(X_test)
df_tcn = evaluate_model('TCN', y_test, tcn_y_pred)

# 1D-CNN
cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    GlobalAveragePooling1D(),
    Dense(len(sensor_cols))
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
cnn_y_pred = cnn_model.predict(X_test)
df_cnn = evaluate_model('1D_CNN', y_test, cnn_y_pred)

# Transformer
def transformer_block(inputs, num_heads=2, ff_dim=64, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = transformer_block(input_layer)
x = GlobalAveragePooling1D()(x)
output_layer = Dense(len(sensor_cols))(x)

transformer_model = Model(inputs=input_layer, outputs=output_layer)
transformer_model.compile(optimizer='adam', loss='mse')
transformer_model.fit(X_train_small, y_train_small, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
transformer_y_pred = transformer_model.predict(X_test)
df_trans = evaluate_model('Transformer', y_test, transformer_y_pred)

# MSE 통합 결과 저장
mse_all = df_gru.merge(df_lstm, on='Sensor')\
                .merge(df_bilstm, on='Sensor')\
                .merge(df_tcn, on='Sensor')\
                .merge(df_cnn, on='Sensor')\
                .merge(df_trans, on='Sensor')

model_names = ['GRU', 'LSTM', 'BiLSTM', 'TCN', '1D_CNN', 'Transformer']

mean_mses = {
    model: mse_all[f'{model}_MSE'].mean()
    for model in model_names
}
sensor_mses = mse_all.to_dict(orient='records')

combined = {
    "summary": mean_mses,
    "details": sensor_mses
}

with open("data/combined_mse.json", "w") as f:
    json.dump(combined, f, indent=2)


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
    "GRU": "gru_trend_model.h5",
    "LSTM": "lstm_trend_model.h5",
    "BiLSTM": "bilstm_trend_model.h5"
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

with open("data/gru_metric.json", "w", encoding="utf-8") as f:
    json.dump(gru_metric_output, f, indent=2, ensure_ascii=False)

# 예측 트렌드 저장 (GRU) - test구간 이후 시점 포함
gru_trend_model = load_model("data/gru_trend_model.h5", compile=False)

X_init = train_data[sensor_cols].values[-60:]
X_window = X_init.copy()

gru_first_y_pred = gru_trend_model.predict(X_window.reshape(1, 60, len(sensor_cols)))
gru_y_pred_rest = gru_trend_model.predict(X_test)
gru_y_pred_full = np.vstack([gru_first_y_pred, gru_y_pred_rest])

future_steps = 10080

future_preds = []
last_input = X_test[-1:].copy()

for _ in range(future_steps):
    pred = gru_trend_model.predict(last_input)
    future_preds.append(pred[0])
    last_input = np.append(last_input[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
future_preds = np.array(future_preds)

train_values_orig = scaler.inverse_transform(train_data[sensor_cols].values)
y_pred_orig = scaler.inverse_transform(gru_y_pred_full)
future_pred_orig = scaler.inverse_transform(future_preds)
y_test_orig = scaler.inverse_transform(y_test)

time_gap = test_data.index[1] - test_data.index[0]
first_pred_time = train_data.index[-1] + time_gap
pred_timestamps = [first_pred_time + i * time_gap for i in range(len(gru_y_pred_full))]

last_pred_time = pred_timestamps[-1] + time_gap
future_timestamps = [last_pred_time + i * time_gap for i in range(future_steps)]

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

