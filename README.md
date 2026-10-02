# FORCORA

### For Customer-Oriented Analysis

FORCORA is an NLP-based customer feedback classification system that automatically analyzes app reviews and categorizes them into actionable feedback types.

Instead of simply determining whether a review is positive or negative, FORCORA focuses on **understanding why a customer is providing feedback**.

The project is being developed as a university minor project with a production-oriented approach, with the long-term goal of building a complete pipeline from **customer review → NLP classification → API → analytics dashboard**.

---

## 📌 Problem Statement

Modern applications receive thousands of customer reviews through platforms such as Google Play Store and Apple App Store.

Manually analyzing these reviews is time-consuming for product teams. More importantly, traditional sentiment analysis such as:

> Positive / Negative / Neutral

does not directly tell developers or product managers what action should be taken.

For example:

> "The application crashes whenever I try to upload a file."

This is not simply a negative review. It represents a **Bug Report**.

Similarly:

> "Please add dark mode."

represents a **Feature Request**.

And:

> "I love this application. It makes my daily routine much easier."

represents **General Feedback**.

FORCORA aims to automatically identify these different types of feedback and convert unstructured customer reviews into structured information that can support product analysis.

---

# 🎯 Project Objective

The primary objective of FORCORA is to develop an NLP-based classification system that automatically categorizes customer reviews into three categories:

| Category | Description | Example |
|---|---|---|
| 🐛 **Bug Report** | Reports a problem, failure, crash, or malfunction | "The app crashes when I upload a file." |
| 💡 **Feature Request** | Requests a new feature or improvement | "Please add dark mode." |
| 💬 **General Feedback** | General comments that do not represent a specific bug or feature request | "Great app, I use it every day." |

The project follows an incremental approach:

1. Explore and preprocess the dataset
2. Build traditional machine-learning baselines
3. Compare different ML models
4. Experiment with a pretrained transformer model
5. Select and package the final model
6. Expose the model through an API
7. Build a web-based analytics dashboard

---

# 🏗️ Overall Project Pipeline

```text
                         CUSTOMER REVIEWS
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Data Collection /   │
                     │ Existing Dataset    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Data Preprocessing  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Exploratory Data    │
                     │ Analysis            │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Train / Test Split  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Text Representation │
                     │      TF-IDF         │
                     └──────────┬──────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │ Multinomial      │    │ Random Forest    │
          │ Naive Bayes      │    │                  │
          └────────┬─────────┘    └────────┬─────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Model Comparison     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transformer Model    │
                    │     DistilBERT       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Model          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ REST API             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Web Dashboard        │
                    └─────────────────────┘
```

---

# 📊 Dataset

The current development dataset consists of English app/customer reviews annotated for three categories.

The original dataset annotations contain:

- `problem_report`
- `feature_request`
- `irrelevant`

For FORCORA, these are mapped to:

```text
problem_report  →  Bug Report
feature_request →  Feature Request
irrelevant      →  General Feedback
```

After preprocessing and removal of ambiguous/unusable samples, the current processed dataset contains:

**5,895 reviews**

### Class Distribution

| Category | Number of Reviews |
|---|---:|
| Bug Report | 1,863 |
| Feature Request | 1,046 |
| General Feedback | 2,986 |
| **Total** | **5,895** |

The dataset is not perfectly balanced, so class-level metrics such as precision, recall, and macro F1 are also considered during evaluation.

---

# 🔍 Exploratory Data Analysis

Before training any model, the dataset is analyzed to understand its structure and quality.

The exploration includes:

- Dataset dimensions
- Column inspection
- Missing values
- Label distribution
- Sample reviews
- Ambiguous labels
- Review length statistics
- Review length distribution

The exploratory analysis is available in:

```text
notebooks/01_dataset_exploration.ipynb
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline converts the raw dataset into a clean dataset that can be consistently used by all machine-learning experiments.

```text
Raw Dataset
     │
     ▼
Load CSV files
     │
     ▼
Combine datasets
     │
     ▼
Check missing / invalid data
     │
     ▼
Remove ambiguous labels
     │
     ▼
Create FORCORA labels
     │
     ▼
Clean review text
     │
     ▼
Save processed dataset
```

The resulting dataset is stored as:

```text
data/processed/forcora_reviews.csv
```

---

# 🤖 Machine Learning Pipeline

## 1. Train/Test Split

The processed dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

A stratified split is used to maintain approximately the same class distribution in both subsets.

The test set is kept separate from model training and is used only for evaluation.

---

# 📝 TF-IDF

Traditional machine-learning algorithms cannot directly process raw text.

FORCORA therefore uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert customer reviews into numerical feature vectors.

Conceptually:

```text
"The app crashes during upload"
              │
              ▼
            TF-IDF
              │
              ▼
     Numerical feature vector
```

The current configuration uses:

- Lowercase conversion
- English stop-word removal
- Unigrams and bigrams
- Maximum feature vocabulary of 10,000

To prevent data leakage:

```text
Training data → fit_transform()
Testing data  → transform()
```

The TF-IDF vectorizer is fitted only on the training data.

---

# 🧠 Model 1 — Multinomial Naive Bayes

Multinomial Naive Bayes is used as the initial baseline model.

The purpose of this model is to establish a reference point for subsequent experiments.

### Results

| Metric | Result |
|---|---:|
| Accuracy | **80.07%** |
| Macro F1 | **0.67** |
| Weighted F1 | **0.76** |

### Class-Level Performance

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bug Report | 0.75 | 0.91 | 0.82 |
| Feature Request | 0.95 | 0.19 | 0.32 |
| General Feedback | 0.83 | 0.94 | 0.88 |

The main weakness of the baseline was **Feature Request recall**.

Although the model had high precision for Feature Requests, it identified only a relatively small proportion of the actual Feature Requests.

This motivated the evaluation of another classical machine-learning algorithm.

---

# 🌲 Model 2 — Random Forest

Random Forest was evaluated using the same processed dataset, train/test split, and TF-IDF representation.

### Results

| Metric | Result |
|---|---:|
| Accuracy | **82.44%** |
| Macro F1 | **0.78** |
| Weighted F1 | **0.82** |

### Class-Level Performance

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bug Report | 0.78 | 0.84 | 0.81 |
| Feature Request | 0.74 | 0.55 | 0.63 |
| General Feedback | 0.87 | 0.91 | 0.89 |

The most significant improvement was in Feature Request classification.

Feature Request recall improved from:

```text
Naive Bayes   → 19%
Random Forest → 55%
```

Feature Request F1 improved from:

```text
Naive Bayes   → 0.32
Random Forest → 0.63
```

This demonstrates why FORCORA evaluates models using multiple metrics rather than relying only on accuracy.

---

# 📈 Current Model Comparison

| Model | Accuracy | Macro F1 | Feature Request Recall |
|---|---:|---:|---:|
| Multinomial Naive Bayes | 80.07% | 0.67 | 19% |
| Random Forest | **82.44%** | **0.78** | **55%** |

These are the current classical ML baseline results.

The project will not assume that the current best classical model is the final model. The next stage is to evaluate a pretrained transformer.

---

# 🤗 Deep Learning — Planned DistilBERT Stage

The next major stage is to experiment with **DistilBERT**, a pretrained transformer-based language model.

The objective is to determine whether contextual language representations can improve customer-feedback classification compared with traditional TF-IDF-based models.

The planned pipeline is:

```text
Customer Review
       │
       ▼
   Tokenizer
       │
       ▼
Token IDs + Attention Mask
       │
       ▼
   DistilBERT
       │
       ▼
Text Representation
       │
       ▼
Classification Layer
       │
       ▼
┌───────────────────────┐
│ Bug Report            │
│ Feature Request       │
│ General Feedback      │
└───────────────────────┘
```

The transformer experiment will be evaluated using the same general evaluation framework as the classical ML models.

The final comparison will include:

```text
Multinomial Naive Bayes
          vs
Random Forest
          vs
DistilBERT
```

---

# 🏭 Planned Production Architecture

The long-term goal is to convert the ML experiments into a production-oriented MVP.

```text
                  CUSTOMER REVIEW
                         │
                         ▼
                ┌─────────────────┐
                │  Web Dashboard  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   REST API      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Processing │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Trained NLP     │
                │ Model           │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Classification       │
              │                      │
              │ Bug Report           │
              │ Feature Request      │
              │ General Feedback     │
              └──────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Analytics       │
                │ Dashboard       │
                └─────────────────┘
```

---

# 🖥️ Planned User Flow

A user will eventually be able to submit a customer review through the dashboard.

Example:

```text
Input:
"The application crashes every time I upload a document."
```

The system will process the review:

```text
Review
  ↓
Frontend
  ↓
Backend API
  ↓
Preprocessing
  ↓
Trained Model
  ↓
Prediction
```

Expected output:

```text
Category: Bug Report
```

For batches of reviews, the dashboard can eventually provide:

- Total review count
- Category distribution
- Bug Report trends
- Feature Request trends
- General Feedback trends
- Example reviews
- Filters and search
- Model confidence

---

# 📁 Repository Structure

```text
forcora/
│
├── data/
│   ├── raw/
│   │   └── classification/
│   │
│   └── processed/
│       └── forcora_reviews.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_naive_bayes.ipynb
│   ├── 04_random_forest.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── inference/
│
├── models/
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Some directories are placeholders for upcoming development stages.

---

# 👥 Team Development

FORCORA is being developed by a team of four.

The project can be divided into the following major workstreams:

### ML / NLP

- Dataset preparation
- Exploratory data analysis
- Text preprocessing
- TF-IDF
- Naive Bayes
- Random Forest
- Transformer experimentation
- Model evaluation
- Model serialization
- Inference pipeline

### Backend

- REST API
- Model loading
- Prediction endpoint
- Input validation
- ML/backend integration
- Error handling

### Frontend

- Dashboard interface
- Review input
- Prediction display
- Analytics visualizations
- Filtering and presentation

### Integration / Deployment

- Frontend-backend integration
- Testing
- Deployment
- Repository organization
- Documentation

The exact division of responsibilities can be adjusted as development progresses.

---

# 🗺️ Development Roadmap

## Phase 1 — Dataset & Exploration

- [x] Obtain classification dataset
- [x] Load and inspect datasets
- [x] Clean and preprocess data
- [x] Create FORCORA labels
- [x] Perform exploratory data analysis

## Phase 2 — Classical ML

- [x] Train/test split
- [x] TF-IDF feature extraction
- [x] Multinomial Naive Bayes
- [x] Random Forest
- [x] Classification reports
- [x] Confusion matrices
- [ ] Consolidated model comparison

## Phase 3 — Deep Learning

- [ ] Learn transformer fundamentals
- [ ] Implement DistilBERT
- [ ] Fine-tune transformer
- [ ] Evaluate transformer
- [ ] Compare against classical ML

## Phase 4 — Production MVP

- [ ] Select final model
- [ ] Save trained model
- [ ] Build inference pipeline
- [ ] Build REST API
- [ ] Build frontend dashboard
- [ ] Connect frontend and backend
- [ ] Add validation and error handling

## Phase 5 — Deployment

- [ ] Testing
- [ ] Deployment
- [ ] API documentation
- [ ] Final architecture
- [ ] Technical documentation
- [ ] Project demonstration

---

# 🛠️ Technology Stack

## Current

**Programming**

- Python

**Data Processing**

- Pandas
- NumPy

**Machine Learning**

- Scikit-learn
- Multinomial Naive Bayes
- Random Forest

**NLP**

- TF-IDF

**Visualization**

- Matplotlib
- Seaborn

**Development**

- Jupyter Notebook
- VS Code
- Git
- GitHub

## Planned

**Deep Learning / NLP**

- PyTorch
- Hugging Face Transformers
- DistilBERT

**Backend**

- FastAPI
- Uvicorn

**Frontend**

- To be finalized by the frontend team

**Deployment**

- To be finalized during the productionization stage

---

# 📏 Evaluation Strategy

FORCORA will evaluate models using multiple metrics.

### Accuracy

Overall percentage of correctly classified reviews.

### Precision

Measures how many reviews predicted as a particular class actually belong to that class.

### Recall

Measures how many of the actual reviews belonging to a class were successfully identified.

### F1 Score

Combines precision and recall.

### Macro F1

Calculates F1 equally across all classes and is useful when class distributions are imbalanced.

### Confusion Matrix

Shows which categories are being confused with one another.

---

# 🚀 Current Status

FORCORA is currently in the **classical machine-learning experimentation stage**.

### Completed

```text
✓ Dataset preparation
✓ Data preprocessing
✓ Exploratory data analysis
✓ Train/test split
✓ TF-IDF feature extraction
✓ Multinomial Naive Bayes
✓ Random Forest
✓ Classification reports
✓ Confusion matrices
```

### Current Best Classical ML Result

```text
Model: Random Forest

Accuracy: 82.44%
Macro F1: 0.78
Feature Request Recall: 55%
```

### Next Milestone

**DistilBERT transformer experiment**

---

# 🔮 Future Scope

The initial MVP can be extended with:

- Automatic review ingestion
- Google Play Store / App Store integration
- Batch review processing
- Sentiment analysis
- Topic extraction
- Priority/severity prediction
- Duplicate issue detection
- Trend detection
- Review search
- Product/version-based analytics
- Multi-language support
- Model monitoring
- Continuous model retraining
- Cloud deployment

These features are considered future scope and are not required for the initial MVP.

---

# 💡 Vision

FORCORA aims to transform unstructured customer feedback into structured, actionable product information.

```text
Thousands of Reviews
        │
        ▼
      FORCORA
        │
        ▼
┌────────────────────────────┐
│ Bug Reports                │
│ Feature Requests           │
│ General Feedback           │
└──────────────┬─────────────┘
               │
               ▼
       Product Analytics
               │
               ▼
       Better Understanding
       of Customer Feedback
```

The project combines:

**Data Engineering + NLP + Machine Learning + Deep Learning + Backend Development + Frontend Development + Deployment**

with the goal of demonstrating an end-to-end machine-learning product rather than a standalone classification notebook.

---

## 📌 Project Status

**🚧 FORCORA is currently under active development.**

**Current milestone:** Classical ML baseline established.

**Next milestone:** DistilBERT evaluation.

---