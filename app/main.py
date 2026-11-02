"""Digit Classifier service with Prometheus metrics."""

import json
import random
import time
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_digits

from app.train import METRICS_PATH, MODEL_PATH

LOW_CONFIDENCE_THRESHOLD = 0.6
TRACKED_PATHS = {"/", "/health", "/model-info", "/sample", "/predict", "/predict-batch"}

# An 8x8 image of the digit 0, row by row
EXAMPLE_PIXELS = [
    0, 0, 5, 13, 9, 1, 0, 0,
    0, 0, 13, 15, 10, 15, 5, 0,
    0, 3, 15, 2, 0, 11, 8, 0,
    0, 4, 12, 0, 0, 8, 8, 0,
    0, 5, 8, 0, 0, 9, 8, 0,
    0, 4, 11, 0, 1, 12, 7, 0,
    0, 2, 14, 5, 10, 12, 0, 0,
    0, 0, 6, 13, 10, 0, 0, 0,
]

# Prometheus metrics
REQUEST_COUNT = Counter(
    "app_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("model_predictions_total", "Number of predictions per digit", ["digit"])
CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of each prediction",
    buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
LOW_CONFIDENCE = Counter(
    "model_low_confidence_total", "Predictions with confidence below the threshold"
)
MODEL_ACCURACY = Gauge("model_test_accuracy", "Test accuracy of the loaded model")
UNUSUAL_PIXELS = Histogram(
    "model_input_unusual_pixel_share",
    "Share of pixels in an input image that are darker than almost all training images at that position",
    buckets=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)

app = FastAPI(
    title="Digit Classifier Service",
    description="Recognizes handwritten digits from 8x8 images and exposes live metrics for Prometheus.",
    version="1.0.0",
)

_model = None
_digits_X, _digits_y = load_digits(return_X_y=True)
# For each of the 64 pixel positions, the value that 99% of the digit images stay below.
# Ink above this level is "unusual" and is a simple sign that the incoming data has changed.
PIXEL_LIMITS = np.percentile(_digits_X, 99, axis=0) + 0.5


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run: python -m app.train")
        _model = joblib.load(MODEL_PATH)
        if METRICS_PATH.exists():
            MODEL_ACCURACY.set(json.loads(METRICS_PATH.read_text())["accuracy"])
    return _model


@app.middleware("http")
async def track_requests(request: Request, call_next):
    path = request.url.path
    if path not in TRACKED_PATHS:
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)


class DigitImage(BaseModel):
    pixels: List[float] = Field(
        ..., min_length=64, max_length=64,
        description="64 pixel values of an 8x8 image, row by row, each from 0 (blank) to 16 (dark)",
    )

    model_config = {"json_schema_extra": {"examples": [{"pixels": EXAMPLE_PIXELS}]}}

    @field_validator("pixels")
    @classmethod
    def check_range(cls, values):
        if any(v < 0 or v > 16 for v in values):
            raise ValueError("Each pixel value must be between 0 and 16")
        return values


class Prediction(BaseModel):
    digit: int
    confidence: float
    low_confidence: bool


def predict_samples(samples):
    model = get_model()
    X = np.array(samples, dtype=float)
    for share in (X > PIXEL_LIMITS).mean(axis=1):
        UNUSUAL_PIXELS.observe(float(share))
    probabilities = model.predict_proba(X)
    results = []
    for probs in probabilities:
        index = int(np.argmax(probs))
        digit = int(model.classes_[index])
        confidence = float(probs[index])
        is_low = confidence < LOW_CONFIDENCE_THRESHOLD

        PREDICTIONS.labels(digit=str(digit)).inc()
        CONFIDENCE.observe(confidence)
        if is_low:
            LOW_CONFIDENCE.inc()

        results.append(Prediction(digit=digit, confidence=round(confidence, 4), low_confidence=is_low))
    return results


@app.get("/")
def root():
    return {"service": "Digit Classifier", "docs": "/docs", "metrics": "/metrics"}


@app.get("/health")
def health():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found")
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found. Run: python -m app.train")
    return json.loads(METRICS_PATH.read_text())


@app.get("/sample")
def sample():
    i = random.randrange(len(_digits_X))
    return {"pixels": _digits_X[i].tolist(), "true_digit": int(_digits_y[i])}


@app.post("/predict", response_model=Prediction)
def predict(image: DigitImage):
    return predict_samples([image.pixels])[0]


@app.post("/predict-batch", response_model=List[Prediction])
def predict_batch(images: List[DigitImage]):
    if not images:
        raise HTTPException(status_code=400, detail="Provide at least one image")
    if len(images) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per request")
    return predict_samples([img.pixels for img in images])


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Load the model at startup so the accuracy metric is available right away
try:
    get_model()
except HTTPException:
    pass
