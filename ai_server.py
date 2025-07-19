# 혼잡도 분류값 변경 
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import joblib
import gdown
import time
import json  # 추가됨

app = FastAPI()

# 모델 및 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 py 파일 기준 경로
MODEL_PATH = os.path.join(BASE_DIR, "congestion_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "congestion_thresholds.json")
DRIVE_URL = "https://drive.google.com/uc?id=13wMzAIHPCo4I_VpWk1GA5K2a60q1X4Fx"

# 입력 데이터 모델
class WeatherInput(BaseModel):
    line: str
    station_name: str
    datetime: str
    direction: int
    TMP: float
    REH: float
    PCP: float
    WSD: float
    SNO: float
    VEC: float

# 전역 모델/기준값 변수
model = None
thresholds = {}

# 혼잡도 등급 분류 함수 (동적 기준 기반)
def categorize_congestion_dynamic(value: float, q1: float, q2: float, q3: float) -> str:
    if value <= q1:
        return "여유"
    elif value <= q2:
        return "보통"
    elif value <= q3:
        return "주의"
    else:
        return "혼잡"

@app.post("/predict")
def predict(data: WeatherInput):
    global model, thresholds
    start_time = time.time()

    # 모델 및 기준값 로드 (최초 1회만)
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                print("모델 파일이 존재하지 않아 다운로드를 시도합니다...")
                gdown.download(DRIVE_URL, MODEL_PATH, fuzzy=True)
                print("모델 다운로드 완료.")
            model = joblib.load(MODEL_PATH)
            print("모델 로드 완료.")
        except Exception as e:
            return {"status": "error", "message": f"모델 로드 실패: {e}"}

    if not thresholds:
        try:
            with open(THRESHOLD_PATH, "r") as f:
                thresholds = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"혼잡도 기준값 로드 실패: {e}"}

    # 날짜 파싱 및 파생 변수 계산
    try:
        dt = datetime.fromisoformat(data.datetime)
        year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
        weekend = int(dt.weekday() >= 5)
        season = ((month % 12 + 3) // 3 - 1)
        Ta = data.TMP
        RH = data.REH / 100
        discomfort = (9/5) * Ta - 0.55 * (1 - RH) * ((9/5) * Ta - 26) + 32
    except Exception as e:
        return {"status": "error", "message": f"날짜 파싱 실패: {e}"}

    # 모델 예측
    try:
        features = [[
            int(data.line), data.direction, data.TMP, data.VEC, data.WSD,
            data.PCP, data.REH, data.SNO,
            year, month, day, hour,
            discomfort, weekend, season
        ]]
        predicted_value = model.predict(features)[0]
        predicted_value = int(round(predicted_value))  # 소수점 제거
    except Exception as e:
        return {"status": "error", "message": f"예측 실패: {e}"}

    level = categorize_congestion_dynamic(
        predicted_value,
        thresholds.get("q1", 80),
        thresholds.get("q2", 130),
        thresholds.get("q3", 150)
    )

    result = {
        "line": data.line,
        "station_name": data.station_name,
        "datetime": data.datetime,
        "direction": data.direction,
        "TMP": data.TMP,
        "REH": data.REH,
        "PCP": data.PCP,
        "WSD": data.WSD,
        "SNO": data.SNO,
        "VEC": data.VEC,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "season": season,
        "weekend": weekend,
        "discomfort": round(discomfort, 2),
        "predicted_congestion_score": predicted_value,
        "predicted_congestion_level": level
    }

    return {
        "status": "ok",
        "congestion_level": level,
        "congestion_score": predicted_value,
        "total_time_sec": round(time.time() - start_time, 3),
        "result": result
    }
