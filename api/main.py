
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model import predict_review
from api.schemas import ReviewRequest, PredictionResponse
from api.bulk import router as bulk_router
from api.database import SessionLocal
from api.models import Review

app = FastAPI(
    title="FORCORA API",
    description="Customer review classification using a trained Random Forest model.",
    version="1.0.0"
)

app.include_router(bulk_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Total-Rows",
        "X-Classified-Rows",
        "X-Skipped-Rows"
    ],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to the FORCORA API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):

    # Predict review using ML model
    result = predict_review(request.review)

    # Save prediction to PostgreSQL
    db = SessionLocal()

    try:
        db_review = Review(
            review_text=request.review,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            review_type="Single Review"
        )

        db.add(db_review)
        db.commit()

    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Could not save review to database."
        )

    finally:
        db.close()

    return result
