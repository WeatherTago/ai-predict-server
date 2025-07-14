FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ai_server.py .
COPY congestion_thresholds.json .

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000"]
