
import csv
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.model import model, LABELS
from api.database import SessionLocal
from api.models import Review

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 10000


@router.post("/predict/bulk")
async def predict_bulk(
    file: UploadFile = File(...),
    review_column: str = Form(...)
):
    # Check file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    # Read file and check size
    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum allowed size is 10 MB."
        )

    # Decode CSV
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV. Please save it using UTF-8 encoding."
        )

    # Parse CSV
    try:
        csv_file = io.StringIO(text)
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty or has no column headers."
            )

        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise HTTPException(
                status_code=400,
                detail="The CSV contains duplicate column names."
            )

        if review_column not in reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{review_column}' was not found. Available columns: {reader.fieldnames}"
            )

        rows = list(reader)

    except csv.Error:
        raise HTTPException(
            status_code=400,
            detail="The CSV file could not be parsed."
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="The CSV contains no review rows."
        )

    if len(rows) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows. Maximum allowed is {MAX_ROWS}."
        )

    # Prepare valid reviews for prediction
    valid_reviews = []
    valid_row_indices = []

    for index, row in enumerate(rows):
        review = row.get(review_column)

        if review and review.strip():
            valid_reviews.append(review.strip())
            valid_row_indices.append(index)

    # Predict all valid reviews together
    predictions = {}

    if valid_reviews:
        predicted_labels = model.predict(valid_reviews)
        class_scores = model.predict_proba(valid_reviews)

        for position, row_index in enumerate(valid_row_indices):
            label = int(predicted_labels[position])
            scores = class_scores[position]

            predictions[row_index] = {
                "predicted_label": label,
                "predicted_category": LABELS[label],
                "confidence": round(
                    float(scores[list(model.classes_).index(label)]), 4
                ),
                "class_scores": {
                    LABELS[int(class_id)]: round(float(score), 4)
                    for class_id, score in zip(model.classes_, scores)
                }
            }

    # Save classified reviews to PostgreSQL
    db = SessionLocal()

    try:
        db_reviews = []

        for row_index in valid_row_indices:
            result = predictions[row_index]

            db_reviews.append(
                Review(
                    review_text=rows[row_index][review_column].strip(),
                    predicted_category=result["predicted_category"],
                    confidence=result["confidence"],
                    review_type="CSV"
                )
            )

        if db_reviews:
            db.add_all(db_reviews)
            db.commit()

        print(f"Saved {len(db_reviews)} CSV reviews to PostgreSQL")

    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Could not save reviews to database."
        )

    finally:
        db.close()

    # Prepare downloadable CSV
    output = io.StringIO(newline="")

    result_columns = [
        "predicted_label",
        "predicted_category",
        "confidence",
        "feature_request_score",
        "bug_report_score",
        "privacy_irrelevant_score",
        "prediction_status"
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=reader.fieldnames + result_columns
    )

    writer.writeheader()

    for index, row in enumerate(rows):
        result = predictions.get(index)

        if result:
            scores = result["class_scores"]

            row.update({
                "predicted_label": result["predicted_label"],
                "predicted_category": result["predicted_category"],
                "confidence": result["confidence"],
                "feature_request_score": scores.get("Feature Request", ""),
                "bug_report_score": scores.get("Bug Report", ""),
                "privacy_irrelevant_score": scores.get("Privacy-Irrelevant", ""),
                "prediction_status": "Classified"
            })

        else:
            row.update({
                "predicted_label": "",
                "predicted_category": "",
                "confidence": "",
                "feature_request_score": "",
                "bug_report_score": "",
                "privacy_irrelevant_score": "",
                "prediction_status": "Skipped - Empty review"
            })

        writer.writerow(row)

    output_bytes = output.getvalue().encode("utf-8-sig")

    total_rows = len(rows)
    classified_rows = len(valid_reviews)
    skipped_rows = total_rows - classified_rows

    return StreamingResponse(
        iter([output_bytes]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="forcora_classified_reviews.csv"',
            "X-Total-Rows": str(total_rows),
            "X-Classified-Rows": str(classified_rows),
            "X-Skipped-Rows": str(skipped_rows)
        }
    )
