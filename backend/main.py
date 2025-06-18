import os
import csv
import io
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse

matplotlib.use("Agg")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../frontend/public/sensor.csv")
JSON_DIR = BASE_DIR

def load_csv():
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV 파일이 존재하지 않음: {CSV_PATH}")
        raise HTTPException(status_code=404, detail="CSV 파일이 존재하지 않습니다.")
    return pd.read_csv(CSV_PATH)

def load_json(filename):
    path = os.path.join(JSON_DIR, filename)
    if not os.path.exists(path):
        logger.error(f"{filename} 파일이 존재하지 않음")
        raise HTTPException(status_code=404, detail=f"{filename} 파일이 존재하지 않습니다.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/headers")
def get_csv_headers():
    try:
        data = load_csv()
        all_zero_cols = data.columns[(data == 0).all()].tolist()
        all_nan_cols = data.columns[data.isna().all()].tolist()
        low_std_cols = data.std(numeric_only=True).loc[lambda x: x < 1.2].index.tolist()
        data.drop(columns=set(all_zero_cols + all_nan_cols + low_std_cols), inplace=True)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)
        data = data.infer_objects(copy=False).interpolate(method="time")
        sensor_cols = [col for col in data.columns if col.startswith("sensor")]
        data[sensor_cols] = StandardScaler().fit_transform(data[sensor_cols])
        data.drop(columns=["machine_status"], inplace=True)
        data.sort_index(inplace=True)
        return {"headers": list(data.columns)[1:]}
    except Exception:
        logger.exception("CSV 헤더 전처리 실패")
        raise HTTPException(status_code=500, detail="CSV 헤더 전처리 실패")

@app.get("/api/timestamp_example")
def get_timestamp_rows():
    try:
        df = load_csv()
        result = df.iloc[0:5, 1]
        return {"headers": [df.columns[1]], "data": [[v] for v in result.tolist()]}
    except Exception:
        logger.exception("Timestamp 샘플 조회 실패")
        raise HTTPException(status_code=500, detail="Timestamp 샘플 조회 실패")

@app.get("/api/sensor_example")
def get_sensor_rows():
    try:
        df = load_csv()
        result = df.iloc[0:5, 2:10].round(5)
        return {"headers": list(result.columns), "data": result.values.tolist()}
    except Exception:
        logger.exception("Sensor 샘플 조회 실패")
        raise HTTPException(status_code=500, detail="Sensor 샘플 조회 실패")

@app.get("/api/status_example")
def get_status_rows():
    try:
        df = load_csv()
        last_column = df.iloc[:, -1]
        result = pd.DataFrame(last_column.dropna().unique(), columns=[df.columns[-1]])
        counts = last_column.value_counts().to_dict()
        return {"headers": list(result.columns), "data": result.values.tolist(), "counts": counts}
    except Exception:
        logger.exception("상태값 샘플 조회 실패")
        raise HTTPException(status_code=500, detail="상태값 샘플 조회 실패")

@app.get("/api/f1-score-svg")
def f1_score_svg():
    try:
        f1_json = load_json("f1_scores.json")
        f1_scores = pd.DataFrame(f1_json["data"], index=f1_json["index"], columns=f1_json["columns"])
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(f1_scores, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
        plt.title("F1-score Comparison by Model and Class")
        plt.tight_layout()
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        return Response(content=buf.getvalue(), media_type="image/svg+xml")
    except Exception:
        logger.exception("F1-score SVG 생성 실패")
        raise HTTPException(status_code=500, detail="F1-score SVG 생성 실패")

@app.get("/api/prediction-svg")
def prediction_svg():
    try:
        data = load_json("prediction_results.json")
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        model_names = ['XGBoost', 'LightGBM', '1D-CNN', 'LSTM']
        fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
        for i, model in enumerate(model_names):
            axes[i].plot(df["timestamp"], df["actual"], color="blue", label="Actual")
            axes[i].plot(df["timestamp"], df[model], color="red", alpha=0.6, label=f"Predicted ({model})")
            axes[i].set_ylabel("Status")
            axes[i].set_title(f"Actual vs Predicted {model}")
            axes[i].grid(True)
            axes[i].legend(loc="upper right")
        plt.xlabel("Timestamp")
        plt.tight_layout()
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        return Response(content=buf.getvalue(), media_type="image/svg+xml")
    except Exception:
        logger.exception("Prediction SVG 생성 실패")
        raise HTTPException(status_code=500, detail="Prediction SVG 생성 실패")

@app.get("/api/mse-barplot-svg")
def mse_barplot_svg():
    try:
        data = load_json("combined_mse.json")
        mean_mses = data["summary"]
        mse_df = pd.DataFrame(mean_mses.items(), columns=["Model", "Avg_MSE"]).sort_values("Avg_MSE")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=mse_df, x="Avg_MSE", y="Model", hue="Model", dodge=False, legend=False, palette="viridis")
        plt.title("모델별 평균 MSE 비교")
        plt.xlabel("Average MSE")
        plt.ylabel("Model")
        plt.tight_layout()
        buf = io.StringIO()
        plt.savefig(buf, format="svg")
        return Response(content=buf.getvalue(), media_type="image/svg+xml")
    except Exception:
        logger.exception("MSE Barplot SVG 생성 실패")
        raise HTTPException(status_code=500, detail="MSE Barplot SVG 생성 실패")

@app.get("/api/model-metrics-summary")
def model_metrics_summary():
    try:
        data = load_json("model_metrics_comparison.json")
        summary = data.get("summary", {})
        rows = [
            {
                "Model": model,
                "Avg_MSE": round(info["Average_MSE"], 5),
                "Avg_MAE": round(info["Average_MAE"], 5),
                "Avg_R2": round(info["Average_R2"], 5)
            } for model, info in summary.items()
        ]
        return JSONResponse(content=pd.DataFrame(rows).sort_values("Avg_MSE").to_dict(orient="records"))
    except Exception:
        logger.exception("모델 메트릭 요약 생성 실패")
        raise HTTPException(status_code=500, detail="모델 메트릭 요약 생성 실패")

@app.get("/api/gru-plot")
def get_gru_forecast(sensor: str = Query(...)):
    try:
        data = load_json("gru_forecast.json")
        matched = next((item for item in data if item["sensor"] == sensor), None)
        if matched is None:
            raise HTTPException(status_code=404, detail=f"{sensor}에 대한 예측 데이터 없음")
        result = {
            "sensor": sensor,
            "train": list(zip(matched["train_timestamps"], matched["train_values"])),
            "test": list(zip(matched["test_timestamps"], matched["actual_values"])),
            "predicted": matched["predicted_values"],
            "actual": matched["actual_values"]
        }
        return JSONResponse(content=result)
    except Exception:
        logger.exception("GRU 예측 데이터 로딩 실패")
        raise HTTPException(status_code=500, detail="GRU 예측 데이터 로딩 실패")

@app.get("/api/recovering-regions")
def get_recovering_regions():
    try:
        data = load_json("recovering_regions.json")
        return JSONResponse(content=data)
    except Exception:
        logger.exception("Recovering Region 로딩 실패")
        raise HTTPException(status_code=500, detail="Recovering Region 로딩 실패")

@app.get("/api/future-recovering")
def get_future_recovering():
    try:
        data = load_csv()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)
        split_idx = int(len(data) * 0.6)
        last_train_time = data.iloc[:split_idx].index[-1]
        raw = load_json("recovering_regions.json")
        timestamps = sorted([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in raw[0]["recovering_timestamps"]])
        firsts = []
        prev = None
        for t in timestamps:
            if prev is None or (t - prev) > timedelta(minutes=60):
                firsts.append(t)
            prev = t
        future = [t.strftime("%Y-%m-%d %H:%M:%S") for t in firsts if t > last_train_time]
        return JSONResponse(content={
            "last_train_timestamp": last_train_time.strftime("%Y-%m-%d %H:%M:%S"),
            "future_recovering": future
        })
    except Exception:
        logger.exception("Future Recovering 예측 실패")
        raise HTTPException(status_code=500, detail="Future Recovering 예측 실패")

@app.get("/api/gru-metrics")
def get_gru_metrics(sensor: str = Query(...)):
    try:
        data = load_json("gru_metrics.json")
        df = pd.DataFrame(data.get("details", []))
        matched = df[df["Sensor"] == sensor]
        if matched.empty:
            raise HTTPException(status_code=404, detail=f"{sensor}에 대한 메트릭 없음")
        r2 = float(matched.iloc[0]["R2"])
        return JSONResponse(content={
            "sensor": sensor,
            "confidence_score": round(r2 * 100, 2)
        })
    except Exception:
        logger.exception("GRU 메트릭 로딩 실패")
        raise HTTPException(status_code=500, detail="GRU 메트릭 로딩 실패")

@app.get("/api/clustering-metrics-summary")
def clustering_metrics_summary():
    try:
        data = load_json("clustering_scores.json")

        rows = []
        for model_name, cluster_info in data.items():
            for config, metrics in cluster_info.items():
                rows.append({
                    "Model": model_name,
                    "Config": config,
                    "Silhouette": round(metrics["silhouette"], 4),
                    "DaviesBouldin": round(metrics["davies_bouldin"], 4)
                })

        df = pd.DataFrame(rows)
        return JSONResponse(content=df.to_dict(orient="records"))

    except Exception:
        logger.exception("클러스터링 메트릭 요약 생성 실패")
        raise HTTPException(status_code=500, detail="클러스터링 메트릭 요약 생성 실패")

@app.get("/api/tsne-cluster-plot")
def tsne_cluster_plot():
    try:
        data = load_json("tsne_cluster_points.json")
        df = pd.DataFrame(data)

        X_2d = df[['tsne_1', 'tsne_2']].values

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
        cluster_nums = [2, 3, 4]

        for i, n in enumerate(cluster_nums):
            axes[i, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=df[f'kmeans_cluster{n}'], cmap='Set1', s=10)
            axes[i, 0].set_title(f"KMeans Clustering (t-SNE, n={n})")
            axes[i, 0].set_xlabel("TSNE 1")
            axes[i, 0].set_ylabel("TSNE 2")
            axes[i, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=df[f'gmm_cluster{n}'], cmap='Set2', s=10)
            axes[i, 1].set_title(f"GMM Clustering (t-SNE, n={n})")
            axes[i, 1].set_xlabel("TSNE 1")
            axes[i, 1].set_ylabel("TSNE 2")
        plt.tight_layout()
        buf = io.StringIO()
        plt.savefig(buf, format='svg')
        plt.close(fig)

        svg_data = buf.getvalue()
        buf.close()
        return Response(content=svg_data, media_type="image/svg+xml")
    except Exception:
        logger.exception("t-SNE 클러스터 플롯 생성 실패")
        raise HTTPException(status_code=500, detail="t-SNE 클러스터 플롯 생성 실패")

@app.get("/api/kmeans-cluster-summary")
def kmeans_cluster_summary():
    try:
        data = load_json("kmeans_cluster_summary.json")
        df = pd.DataFrame(data)
        return JSONResponse(content=df.to_dict(orient="records"))

    except Exception:
        logger.exception("KMeans 클러스터 요약 데이터 로드 실패")
        raise HTTPException(status_code=500, detail="KMeans 클러스터 요약 데이터 로드 실패")


