
from pathlib import Path
import joblib

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Saved model location
MODEL_PATH = PROJECT_DIR / "models" / "forcora_random_forest.joblib"

# Dataset-specific label mapping
LABELS = {
    0: "Feature Request",
    1: "Bug Report",
    2: "Privacy-Irrelevant"
}

# Load the trained pipeline once
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def predict_review(review: str) -> dict:
    """Predict the category of a customer review."""

    review = review.strip()

    if not review:
        raise ValueError("Review cannot be empty.")

    prediction = int(model.predict([review])[0])

    # Random Forest vote proportions, not calibrated probabilities
    scores = model.predict_proba([review])[0]

    return {
        "predicted_label": prediction,
        "predicted_category": LABELS[prediction],
        "confidence": round(float(scores[prediction]), 4),
        "class_scores": {
            LABELS[int(class_id)]: round(float(score), 4)
            for class_id, score in zip(
                model.classes_,
                scores
            )
        }
    }
