from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.model import predict_review
from api.schemas import ReviewRequest, PredictionResponse


app = FastAPI(
    title="FORCORA API",
    description="Customer review classification using a trained Random Forest model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return predict_review(request.review)
