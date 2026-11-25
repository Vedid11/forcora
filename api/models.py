
from sqlalchemy import Column, Integer, Text, String, Float, DateTime
from sqlalchemy.sql import func
from api.database import Base


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    review_text = Column(Text, nullable=False)
    predicted_category = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=True)
    review_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
