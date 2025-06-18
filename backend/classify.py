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
from sklearn.manifold import TSNE

# 데이터 전처리
df = pd.read_csv('../frontend/public/sensor.csv')

df.interpolate(method='linear', inplace=True)
df = df.drop(columns=['sensor_15'])
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
df_sensors = df[sensor_cols]

sensor_stds = df_sensors.std()
low_std_cols = sensor_stds[sensor_stds < 1.2].index.tolist()
df= df.drop(columns=low_std_cols)
num_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# 고장 유형 분류 window 생성
def extract_pre_failure_windows(broken_times, df, window=10):
    segments = []
    used_broken_times = []

    for i, t in enumerate(broken_times):
        temp_normal = df[df['timestamp'] < t].tail(window)

        temp_broken = df[df['timestamp'] == t]

        if len(temp_normal) == window and len(temp_broken) == 1 and 'BROKEN' not in temp_normal['machine_status'].values:
            combined = pd.concat([temp_normal, temp_broken])
            combined = combined.copy()
            combined['label'] = '1'
            combined['case_id'] = f"case_{i}"
            segments.append(combined)
            used_broken_times.append(t)

    return pd.concat(segments).reset_index(drop=True), used_broken_times

# 데이터 분할 학습(60%-5건), 검증(40%-2건)
broken_times = df[df['machine_status'] == 'BROKEN']['timestamp'].drop_duplicates().sort_values().reset_index(drop=True)
df_pre_failure_all, used_broken_times = extract_pre_failure_windows(broken_times, df)


train_cases = [f"case_{i}" for i in range(5)]       #60%
test_cases = [f"case_{i}" for i in range(5, 7)]     #40%

df_train = df_pre_failure_all[df_pre_failure_all['case_id'].isin(train_cases)].copy()
df_test = df_pre_failure_all[df_pre_failure_all['case_id'].isin(test_cases)].copy()

# 클러스터링 수행
sensor_cols = [col for col in df_train.columns if 'sensor' in col]

X_train = df_train[sensor_cols]
X_test = df_test[sensor_cols]

from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=2, random_state=42)
kmeans2.fit(X_train)
df_train['kmeans_cluster2'] = kmeans2.labels_

kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(X_train)
df_train['kmeans_cluster3'] = kmeans3.labels_

kmeans4= KMeans(n_clusters=4, random_state=42)
kmeans4.fit(X_train)
df_train['kmeans_cluster4'] = kmeans4.labels_

# (2) Gaussian Mixture Model (GMM
from sklearn.mixture import GaussianMixture

gmm2 = GaussianMixture(n_components=2, random_state=42)
gmm2.fit(X_train)
df_train['gmm_cluster2'] = gmm2.predict(X_train)

gmm3 = GaussianMixture(n_components=3, random_state=42)
gmm3.fit(X_train)
df_train['gmm_cluster3'] = gmm3.predict(X_train)

gmm4 = GaussianMixture(n_components=4, random_state=42)
gmm4.fit(X_train)
df_train['gmm_cluster4'] = gmm4.predict(X_train)

cluster_cols = [
    'kmeans_cluster2', 'kmeans_cluster3', 'kmeans_cluster4',
    'gmm_cluster2', 'gmm_cluster3', 'gmm_cluster4'
]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_train)

df_train['tsne_1'] = X_2d[:, 0]
df_train['tsne_2'] = X_2d[:, 1]

save_cols = ['tsne_1', 'tsne_2'] + cluster_cols
tsne_cluster_data = df_train[save_cols].to_dict(orient='records')

with open('data/tsne_cluster_points.json', 'w', encoding='utf-8') as f:
    json.dump(tsne_cluster_data, f, indent=2, ensure_ascii=False)

# 모델 평가
from sklearn.metrics import silhouette_score, davies_bouldin_score

kmeans2_silhouette = silhouette_score(X_train, df_train['kmeans_cluster2'])
kmeans2_dbi = davies_bouldin_score(X_train, df_train['kmeans_cluster2'])

kmeans3_silhouette = silhouette_score(X_train, df_train['kmeans_cluster3'])
kmeans3_dbi = davies_bouldin_score(X_train, df_train['kmeans_cluster3'])

kmeans4_silhouette = silhouette_score(X_train, df_train['kmeans_cluster4'])
kmeans4_dbi = davies_bouldin_score(X_train, df_train['kmeans_cluster4'])

gmm2_silhouette = silhouette_score(X_train, df_train['gmm_cluster2'])
gmm2_dbi = davies_bouldin_score(X_train, df_train['gmm_cluster2'])

gmm3_silhouette = silhouette_score(X_train, df_train['gmm_cluster3'])
gmm3_dbi = davies_bouldin_score(X_train, df_train['gmm_cluster3'])

gmm4_silhouette = silhouette_score(X_train, df_train['gmm_cluster4'])
gmm4_dbi = davies_bouldin_score(X_train, df_train['gmm_cluster4'])

clustering_scores = {
    "KMeans": {
        "n_clusters=2": {
            "silhouette": round(kmeans2_silhouette, 4),
            "davies_bouldin": round(kmeans2_dbi, 4)
        },
        "n_clusters=3": {
            "silhouette": round(kmeans3_silhouette, 4),
            "davies_bouldin": round(kmeans3_dbi, 4)
        },
        "n_clusters=4": {
            "silhouette": round(kmeans4_silhouette, 4),
            "davies_bouldin": round(kmeans4_dbi, 4)
        }
    },
    "GMM": {
        "n_components=2": {
            "silhouette": round(gmm2_silhouette, 4),
            "davies_bouldin": round(gmm2_dbi, 4)
        },
        "n_components=3": {
            "silhouette": round(gmm3_silhouette, 4),
            "davies_bouldin": round(gmm3_dbi, 4)
        },
        "n_components=4": {
            "silhouette": round(gmm4_silhouette, 4),
            "davies_bouldin": round(gmm4_dbi, 4)
        }
    }
}

with open('data/clustering_scores.json', 'w', encoding='utf-8') as f:
    json.dump(clustering_scores, f, indent=2, ensure_ascii=False)

# Summay 저장
df_test['kmeans_cluster3'] = kmeans3.predict(X_test)
train_distances = kmeans3.transform(X_train)
threshold = train_distances.min(axis=1).max()
test_distances = kmeans3.transform(X_test)
df_test['is_new_cluster'] = test_distances.min(axis=1) > threshold

latest_timestamps = df_test.groupby('case_id')['timestamp'].max().reset_index()

summary = (
    df_test.groupby('case_id')
    .agg(
        total=('is_new_cluster', 'size'),
        true_count=('is_new_cluster', 'sum'),
        cluster_mode=('kmeans_cluster3', lambda x: x.mode()[0] if not x.mode().empty else -1)
    )
    .reset_index()
)

summary['false_count'] = summary['total'] - summary['true_count']
summary['is_new_cluster'] = summary['true_count'] >= summary['false_count']
summary['is_new_cluster_highpercent'] = summary.apply(
    lambda row: f"{row['true_count'] / row['total'] * 100:.0f}%", axis=1
)

summary = summary.merge(latest_timestamps, on='case_id')

summary = summary[['case_id', 'cluster_mode', 'is_new_cluster', 'is_new_cluster_highpercent', 'timestamp']]
summary.columns = ['case_id', 'kmeans_cluster3', 'is_new_cluster', 'is_new_cluster_highpercent', 'timestamp']

result_json = summary.to_dict(orient='records')
with open('data/kmeans_cluster_summary.json', 'w', encoding='utf-8') as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)
