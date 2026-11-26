# FORCORA
### For Customer-Oriented Analysis

**An NLP-powered customer feedback classification system for privacy-related social media application reviews.**

[![Live Application](https://img.shields.io/badge/Live%20Application-Cloudflare%20Pages-blue)](https://forcora.pages.dev)
[![Backend API](https://img.shields.io/badge/API-Render-green)](https://forcora-api.onrender.com/docs)
[![Database](https://img.shields.io/badge/Database-Neon%20PostgreSQL-00E599)](https://neon.tech)

**Live application:** https://forcora.pages.dev  
**API documentation:** https://forcora-api.onrender.com/docs  
**GitHub repository:** https://github.com/Vedid11/forcora

---

## 1. Project Overview

FORCORA (For Customer-Oriented Analysis) is a machine-learning-based application designed to classify customer feedback into meaningful categories.

Rather than performing conventional positive/negative sentiment analysis, FORCORA focuses on identifying the intent behind a review.

The current implementation is demonstrated using privacy-related feedback from social media applications.

The system supports individual review classification and bulk CSV classification through a web interface. Predictions are processed by a FastAPI backend and stored in a PostgreSQL database.

## 2. Problem Statement

Mobile and social media applications receive large volumes of customer reviews. Manually examining these reviews and identifying actionable feedback is time-consuming.

Traditional sentiment analysis does not necessarily distinguish between a requested feature, a reported technical issue, and feedback unrelated to the target categories.

FORCORA addresses this problem by classifying reviews into three categories that can help organize customer feedback for further analysis.

## 3. Project Objectives

- Develop an NLP-based customer review classification system.
- Explore and preprocess an annotated review dataset.
- Train and evaluate traditional machine learning models.
- Experiment with a transformer-based classification model.
- Compare models using multiple evaluation metrics.
- Integrate the selected production model with a REST API.
- Provide single-review and bulk CSV classification.
- Store classification results in a PostgreSQL database.
- Deploy the application using cloud hosting services.

## 4. Classification Categories

FORCORA uses the following dataset-specific labels:

| Label | Category | Description |
|---|---|---|
| 0 | Feature Request | Requests for new features or improvements |
| 1 | Bug Report | Reports of errors, failures, or technical problems |
| 2 | Privacy-Irrelevant | Reviews not relevant to the target privacy-related feedback categories |

**Important:** Privacy-Irrelevant is the actual third dataset category. It should not be interpreted as a general positive-feedback category.

## 5. Dataset

The project uses the annotated dataset:

**Annotated Dataset for User Review Classification in the Privacy Domain for Social Media Applications**

- Authors: Farah, Labiba; Kabir, Mohammad Ridwan; Ahmed, Shohel
- Published: 10 July 2025
- Dataset repository: https://zenodo.org/records/15852840
- DOI: https://doi.org/10.5281/zenodo.15852840
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Language: English

### Dataset splits

The processed dataset used in the primary experiments contains 15,945 reviews.

| Dataset split | Number of reviews |
|---|---:|
| Training | 12,756 |
| Validation | 1,595 |
| Testing | 1,594 |
| **Total** | **15,945** |

The dataset contains the following columns:

- `Review`: Original review text
- `Processed`: Processed review text
- `Label`: Assigned classification label

The primary classical machine learning experiments use the original `Review` text.

### Class distribution

| Category | Training | Validation | Testing |
|---|---:|---:|---:|
| Feature Request | 2,878 | 360 | 359 |
| Bug Report | 3,366 | 421 | 420 |
| Privacy-Irrelevant | 6,512 | 814 | 815 |
| **Total** | **12,756** | **1,595** | **1,594** |

The dataset is imbalanced, with Privacy-Irrelevant being the largest category. Therefore, macro-averaged evaluation metrics are considered alongside accuracy.

## 6. Data Exploration and Preprocessing

The dataset inspection and preparation process includes:

- Dataset structure and column inspection
- Missing-value checks
- Class distribution analysis
- Duplicate review inspection
- Cross-split overlap checks
- Review text inspection
- Exploratory keyword and error analysis

The original training, validation, and testing splits are retained for the experiments.

The test set is reserved for final evaluation after model selection.

## 7. Machine Learning Experiments

### 7.1 TF-IDF Feature Extraction

The traditional machine learning models use TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical feature representations.

The primary configuration includes:

- Maximum features: 10,000
- N-gram range: Unigrams and bigrams (1, 2)
- Minimum document frequency: 2
- Sublinear term frequency scaling: Enabled

The vectorizer is fitted on training data and then applied to validation and test data.

### 7.2 Multinomial Naive Bayes

Multinomial Naive Bayes was implemented as the initial classical machine learning baseline.

**Validation results:**

| Metric | Score |
|---|---:|
| Accuracy | 84.39% |
| Macro Precision | 0.8424 |
| Macro Recall | 0.8195 |
| Macro F1 | 0.8296 |

### 7.3 Random Forest

Random Forest was evaluated using the TF-IDF representation.

The configuration uses:

- Number of estimators: 200
- Maximum depth: None
- Class weighting: Balanced
- Random state: 42
- Parallel processing: Enabled

**Validation results:**

| Metric | Score |
|---|---:|
| Accuracy | 84.64% |
| Macro Precision | 0.8364 |
| Macro Recall | 0.8327 |
| Macro F1 | 0.8344 |

Random Forest performed marginally better than Naive Bayes in validation accuracy, recall, and macro F1, while Naive Bayes achieved higher macro precision.

Random Forest was selected for the production application.

### 7.4 DistilBERT Experiment

A transformer-based experiment using DistilBERT was implemented to investigate whether contextual language representations could improve classification performance.

The experiment is maintained separately under:

`bert/distilbert/`

**Test results:**

| Metric | Score |
|---|---:|
| Accuracy | 88.27% |
| Macro Precision | 0.8701 |
| Macro Recall | 0.8732 |
| Macro F1 | 0.8716 |

### 7.5 Final Model Comparison

The following table presents the final test results for Random Forest and DistilBERT, along with the validation results for the Naive Bayes baseline.

| Model | Evaluation split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Multinomial Naive Bayes | Validation | 84.39% | 0.8296 |
| Random Forest | Test | 83.19% | 0.8152 |
| DistilBERT | Test | 88.27% | 0.8716 |

**Interpretation:**

DistilBERT achieved the strongest reported test performance, with a 5.08 percentage-point accuracy improvement and a 0.0564 improvement in macro F1 over Random Forest.

The same test CSV file was verified for the Random Forest and DistilBERT evaluations. However, the experiments use separate implementations, so preprocessing and evaluation details should be considered when interpreting the comparison.

**Production model:** Random Forest.

DistilBERT remains a separate experimental model and is not currently connected to the deployed API.

## 8. Application Architecture

FORCORA follows a frontend–backend–database architecture.

```text
                 USER
                  |
                  v
          CLOUDFLARE PAGES
            Web Frontend
                  |
                  v
             FASTAPI API
               (Render)
                  |
                  v
          RANDOM FOREST MODEL
          TF-IDF + Classifier
                  |
                  v
          CLASSIFICATION RESULT
                  |
                  v
          NEON POSTGRESQL
          Review Record Storage
```

### Application workflow

1. The user submits an individual review or uploads a CSV file.
2. The frontend sends the request to the FastAPI backend.
3. The backend validates the input.
4. The trained Random Forest pipeline processes the review.
5. The predicted category and class scores are generated.
6. The prediction is returned to the frontend.
7. The classified review is stored in PostgreSQL.

## 9. Application Features

### Single Review Classification

- Accepts an individual review.
- Sends the review to the backend API.
- Returns the predicted category.
- Displays classification scores.

### Bulk CSV Classification

- Accepts CSV file uploads.
- Allows selection of the review text column.
- Classifies valid review rows.
- Produces a downloadable classified CSV.
- Includes predicted labels, categories, class scores, and prediction status.

The backend currently enforces a maximum upload size of 10 MB and a maximum of 10,000 data rows.

### Database Integration

Classification records are stored in PostgreSQL.

The `reviews` table includes:

| Column | Description |
|---|---|
| `id` | Record identifier |
| `review_text` | Submitted review |
| `predicted_category` | Predicted category |
| `confidence` | Predicted class vote proportion |
| `review_type` | Single Review or CSV |
| `created_at` | Record creation timestamp |

The Random Forest class scores are based on tree vote proportions and should not be interpreted as calibrated probabilities.

## 10. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Traditional ML | Scikit-learn |
| NLP feature extraction | TF-IDF |
| Transformer experiment | DistilBERT |
| Backend framework | FastAPI |
| API server | Uvicorn |
| Input validation | Pydantic |
| Database | PostgreSQL |
| Database hosting | Neon |
| ORM | SQLAlchemy |
| Model serialization | Joblib |
| Version control | Git and GitHub |
| Model file management | Git LFS |
| Frontend hosting | Cloudflare Pages |
| Backend hosting | Render |

## 11. Deployment

The application is deployed using three services.

| Component | Platform | Deployment |
|---|---|---|
| Frontend | Cloudflare Pages | https://forcora.pages.dev |
| Backend API | Render | https://forcora-api.onrender.com |
| Database | Neon PostgreSQL | Hosted PostgreSQL |

### Live Links

- **Application:** https://forcora.pages.dev
- **API documentation:** https://forcora-api.onrender.com/docs
- **Backend health endpoint:** https://forcora-api.onrender.com/health
- **Source code:** https://github.com/Vedid11/forcora

The backend uses environment variables for database configuration and frontend CORS origins. Database credentials are not stored in the repository.

The Random Forest pipeline is stored under `models/` and managed using Git LFS.

## 12. API Endpoints

The FastAPI application exposes the following main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Classify an individual review |
| POST | `/predict/bulk` | Classify reviews from a CSV upload |

Interactive API documentation is available at:

https://forcora-api.onrender.com/docs

### Example single-review request

```json
{
  "review": "Please add an option to hide my online status"
}
```

### Example response

```json
{
  "predicted_label": 0,
  "predicted_category": "Feature Request",
  "confidence": 0.775,
  "class_scores": {
    "Feature Request": 0.775,
    "Bug Report": 0.025,
    "Privacy-Irrelevant": 0.2
  }
}
```

The example scores illustrate the response structure; actual results depend on the input review.

## 13. Repository Structure

```text
forcora/
│
├── api/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   ├── bulk.py
│   ├── database.py
│   ├── models.py
│   └── test_db.py
│
├── bert/
│   └── distilbert/
│       ├── src/
│       ├── DATA/
│       ├── outputs/
│       ├── run_train.py
│       ├── eval_test.py
│       └── README.md
│
├── data/
│   ├── raw/
│   │   └── privacy_reviews/
│   ├── archive/
│   └── sample_uploads/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   └── forcora_random_forest.joblib
│
├── notebooks/
│   ├── 06_privacy_dataset_inspection.ipynb
│   ├── 07_naivebayes_new_dataset.ipynb
│   ├── 08_random_forest_new_dataset.ipynb
│   ├── 09_model_comparison_new_dataset.ipynb
│   └── 10_error_analysis_new_dataset.ipynb
│
├── requirements.txt
├── requirements-render.txt
├── .gitattributes
├── .gitignore
└── README.md
```

## 14. Local Development Setup

### Prerequisites

- Python 3.11
- Git
- Git LFS
- PostgreSQL or a compatible hosted PostgreSQL database

### Clone the repository

```bash
git clone https://github.com/Vedid11/forcora.git
cd forcora
git lfs pull
```

### Create a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```cmd
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root.

For local PostgreSQL development:

```env
DB_PASSWORD=your_local_database_password
```

For a hosted PostgreSQL database, configure:

```env
DATABASE_URL=your_postgresql_connection_string
```

Never commit `.env` or expose database credentials.

### Start the backend

From the project root:

```cmd
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

The local API will be available at:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

The API initializes the database table during application startup.

## 15. Limitations

- The current dataset focuses on privacy-related feedback from social media applications. Generalization to unrelated customer-feedback domains has not yet been established.
- Classification quality depends on the coverage and annotation quality of the training dataset.
- Some reviews may contain ambiguous or overlapping technical and privacy-related language.
- The deployed API currently uses Random Forest rather than DistilBERT.
- Random Forest class vote proportions are not calibrated confidence probabilities.
- Render's free service may spin down after inactivity, resulting in slower initial requests.
- The current application provides classification and record storage; broader cross-domain analytics and trend analysis remain potential future extensions.

## 16. Future Scope

Possible future improvements include:

- Integrating DistilBERT into the production API.
- Evaluating the system on additional customer-feedback domains.
- Expanding and improving the annotated dataset.
- Adding more advanced review analytics and visualizations.
- Introducing trend analysis and time-based reporting.
- Improving model confidence calibration.
- Adding authentication and user-specific access controls.
- Improving deployment scalability and monitoring.

## 17. Conclusion

FORCORA demonstrates an end-to-end machine learning application for classifying privacy-related social media application reviews.

The project covers dataset inspection, preprocessing, traditional machine learning, transformer experimentation, model evaluation, API development, frontend integration, database storage, and cloud deployment.

The deployed application supports individual and bulk review classification, while storing prediction records in a hosted PostgreSQL database.

The experimental results demonstrate the potential of transformer-based models to improve text classification performance, while the Random Forest model provides the current production implementation.

FORCORA establishes a working foundation for organizing unstructured customer feedback into structured, actionable categories.

## 18. Dataset Reference

Farah, L., Kabir, M. R., & Ahmed, S. (2025). *Annotated Dataset for User Review Classification in the Privacy Domain for Social Media Applications*. Zenodo.

https://doi.org/10.5281/zenodo.15852840

Dataset license: CC BY 4.0. Appropriate attribution is required when redistributing or using the dataset.

---

**FORCORA — For Customer-Oriented Analysis**  
*Turning unstructured customer feedback into structured insights.*