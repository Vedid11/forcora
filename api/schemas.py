
from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    review: str = Field(
        min_length=1,
        max_length=5000,
        description="Customer review text to classify"
    )

    @field_validator("review")
    @classmethod
    def validate_review(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Review cannot be empty.")

        return value


class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_category: str
    confidence: float
    class_scores: dict[str, float]
