import csv
import os
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from datetime import datetime

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

def get_csv_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "frontend", "public", "sensor.csv")
    if not os.path.exists(path):
        logger.error(f"CSV file not found at path: {path}")
        raise HTTPException(status_code=404, detail="CSV file not found")
    return path

@app.get("/api/headers")
def get_csv_headers():
    try:
        with open(get_csv_path(), newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            return {"headers": headers[2:]}
    except Exception as e:
        logger.exception("Failed to read CSV headers")
        raise HTTPException(status_code=500, detail="Failed to read CSV headers")

@app.get("/api/sensor-data")
def get_sensor_data(
    sensor: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None)
):
    timestamps = []
    values = []
    try:
        with open(get_csv_path(), newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if sensor not in reader.fieldnames:
                return {"error": f"{sensor} not found in CSV"}

            for row in reader:
                ts_str = row.get("timestamp")
                val = row.get(sensor)
                if not ts_str or not val:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except:
                    continue
                if start:
                    try:
                        if ts < datetime.fromisoformat(start):
                            continue
                    except:
                        pass
                if end:
                    try:
                        if ts > datetime.fromisoformat(end):
                            continue
                    except:
                        pass
                timestamps.append(ts.timestamp())
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(None)
        return {"timestamps": timestamps, "values": values}
    except Exception as e:
        logger.exception("Failed to read CSV headers")
        raise HTTPException(status_code=500, detail="Failed to read CSV headers")

@app.get("/api/raw-data")
def get_raw_data(
    page: int = Query(1, gt=0),
    size: int = Query(20, gt=0)
):
    try:
        with open(get_csv_path(), newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            total = len(all_rows)

            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated = all_rows[start_idx:end_idx]

            return {
                "data": paginated,
                "total": total,
                "page": page,
                "size": size
            }
    except Exception as e:
        logger.exception("Failed to fetch paginated raw data")
        raise HTTPException(status_code=500, detail="Failed to fetch raw data")