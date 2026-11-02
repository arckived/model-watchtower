"""Train a handwritten digit classifier and save it with its evaluation metrics."""

import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from app.preprocessing import scale_pixels

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_model(random_state=42):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", FunctionTransformer(scale_pixels)),
        ("classifier", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "model": "SVC (RBF kernel) with calibrated probabilities",
        "dataset": "scikit-learn digits (8x8 images)",
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    results = train_model()
    print("Model saved to", MODEL_PATH)
    print(json.dumps(results, indent=2))
