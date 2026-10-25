# Workplace Passive-Aggression Detection API

Lightweight, production-ready FastAPI microservice exposing the workplace passive-aggression classification model from `abijitabi47/passive-aggression-detector`.

## Architecture
- **Classifier:** Scikit-Learn TF-IDF (1-2 n-grams, sublinear TF) + Logistic Regression (class_weight='balanced')
- **Supporting Signal:** VADER Sentiment compound polarity
- **Explainability:** Exact coefficient-weighted n-gram cue terms
- **Pre-trained Artifact:** `models/passive_aggression_pipeline.joblib` (Loaded into memory once at service startup)

## Endpoints

### 1. `GET /health`
Returns service readiness and confirms model is loaded in memory:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "workplace_logistic_regression_v1",
  "version": "1.0.0",
  "classes": [
    "Not flagged as passive-aggressive",
    "Potentially passive-aggressive"
  ]
}
```

### 2. `POST /predict`
Analyzes a single workplace statement:
**Request:**
```json
{
  "text": "Per my previous email, as I already explained twice yesterday."
}
```

**Response:**
```json
{
  "model": "workplace_logistic_regression_v1",
  "text": "Per my previous email, as I already explained twice yesterday.",
  "prediction": 1,
  "label": "Potentially passive-aggressive",
  "score": 0.9489,
  "sentiment_label": "Positive",
  "sentiment_score": 0.0,
  "cues": ["previous email", "explained", "twice"],
  "latency_ms": 2.14
}
```

### 3. `GET /docs`
Interactive Swagger / OpenAPI UI for test calls.

---

## Local Execution

```bash
# From workspace root:
uvicorn workplace_api.main:app --host 0.0.0.0 --port 8001 --reload
```

Test with curl:
```bash
curl -X POST http://localhost:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Thanks for ignoring my message again."}'
```

---

## Free Cloud Deployment (Render.com)

1. Connect your GitHub repository to Render (https://render.com).
2. Select **New Web Service** -> select your repo.
3. Set:
   - **Root Directory:** `workplace_api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/health`
   - **Plan:** Free
4. Deploy! Your API will be live at `https://<your-render-app>.onrender.com`.
