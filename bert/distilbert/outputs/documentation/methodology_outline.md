# FORCORA: Methodology & Empirical Evaluation Outline
**Automated Classification of Privacy-Related User Reviews Using DistilBERT**

---

## 1. Problem Statement

Mobile application users frequently communicate issues, suggestions, and feedback concerning privacy practices, permissions, and data security directly through app store user reviews. Manually inspecting large streams of unstructured user feedback is labor-intensive and unscalable for software engineering and privacy compliance teams. 

The **FORCORA** project establishes an automated text classification pipeline designed to categorize privacy-relevant and general app store reviews into actionable software maintenance categories. Specifically, reviews are mapped to three discrete classes:
- **Feature Requests**: User requests for new privacy controls, policy opt-outs, or functionality enhancements.
- **Bug Reports**: User reports detailing defects, unexpected tracking, permission failures, or privacy violations.
- **Privacy-Irrelevant Feedback**: User reviews that either do not concern privacy or represent general operational app reviews lacking privacy-specific software maintenance signals.

The primary objective is to develop, rigorously fine-tune, and empirically benchmark a transformer-based sequence classifier (`distilbert-base-uncased`) against traditional machine learning baselines under strict dataset isolation protocols.

---

## 2. Dataset Description

The experimental corpus comprises **15,945** annotated user reviews situated in the privacy reviews domain.

### 2.1 Corpus Characteristics
- **Total Samples**: 15,945 reviews across three pre-partitioned CSV files.
- **Columns Available**:
  - `Review`: Raw, uncleaned natural language text reflecting exact user submissions (punctuation, casing, emojis, and whitespace intact).
  - `Processed`: Lemmatized, lower-cased, and punctuation-stripped text representations.
  - `Label`: Integer ground-truth classification target $\in \{0, 1, 2\}$.
- **Missing Values**: 0 across all files and columns (100% complete).

### 2.2 Sequence Length & Token Distribution
Exact tokenization diagnostics conducted using the `distilbert-base-uncased` WordPiece tokenizer (with special tokens `[CLS]` and `[SEP]`) on the raw `Review` field yielded the following sequence length properties:
- **Minimum Tokens**: 3 (including `[CLS]` and `[SEP]`)
- **Median Tokens**: ~23 tokens
- **Mean Tokens**: ~33.8 tokens
- **95th Percentile**: 97 tokens
- **99th Percentile**: 198 tokens
- **Maximum Tokens**: 731 tokens (isolated extreme outlier)
- **Coverage at Sequence Length 256**: **99.78%** of all samples require zero truncation. Only 0.22% (35 samples total across the corpus) exceed 256 tokens.

---

## 3. Label Definitions & Class Distribution

The target variable is partitioned into three mutually exclusive categories:

| Class ID | Label Name | Operational Definition | Total Support | Total Share |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Feature Request** | User suggests additions or changes to privacy mechanisms, permission settings, or user controls. | 3,594 | 22.54% |
| **1** | **Bug Report** | User identifies functional defects, broken privacy protections, data leaks, or crashes related to privacy permissions. | 4,204 | 26.37% |
| **2** | **Privacy-Irrelevant** | User text does not concern privacy policies/features or provides generic non-privacy sentiment. | 8,147 | 51.09% |
| **Total** | — | — | **15,945** | **100.00%** |

The distribution reflects natural class imbalance common in app store corpora, with majority background noise (Privacy-Irrelevant at ~51.1%) and specialized actionable signals (Feature Requests at ~22.5% and Bug Reports at ~26.4%).

---

## 4. Data Splitting & Leakage Prevention

The dataset is partitioned into three disjoint splits strictly maintained in `G:\FORCORA\DATA\RAW\privacy_reviews\`:

| Split Name | Filename | Row Count | Percentage | Class 0 (FR) | Class 1 (BR) | Class 2 (PI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `original_12756 (1).csv` | 12,756 | 80.00% | 2,876 (22.55%) | 3,363 (26.36%) | 6,517 (51.09%) |
| **Validation** | `val_1595.csv` | 1,595 | 10.00% | 359 (22.51%) | 421 (26.39%) | 815 (51.10%) |
| **Test** | `test_1594.csv` | 1,594 | 9.997% | 359 (22.52%) | 420 (26.35%) | 815 (51.13%) |
| **Total** | — | **15,945** | **100.00%** | **3,594** | **4,204** | **8,147** |

### Leakage Prevention Guarantees:
1. **Pre-partitioned Isolation**: Splits were generated upstream and preserved without post-hoc resplitting or leakage across boundaries.
2. **Strict Test Isolation**: The test dataset (`test_1594.csv`) was never loaded, sampled, or accessed during hyperparameter tuning, model development, loss computation, or checkpoint selection.
3. **Single Evaluation Protocol**: Test evaluation was executed exactly once following the completion and freeze of model training.

---

## 5. TF-IDF Baseline Methodology & Audit Status

### 5.1 Recovered Artifact Inspection
An external serialized model artifact was located at `C:\Users\aryan\Downloads\forcora_random_forest.joblib` (117.3 MB, timestamp 2026-11-24 00:52:37). Inspection of the pickle metadata reveals the following pipeline architecture:
- **Container**: `sklearn.pipeline.Pipeline`
- **Step 1 (`tfidf`)**: `TfidfVectorizer`
  - `analyzer`: `'word'`
  - `ngram_range`: `(1, 1)` (unigrams)
  - `lowercase`: `True`
  - `norm`: `'l2'`
  - `use_idf`: `True`, `smooth_idf`: `True`, `sublinear_tf`: `False`
  - `token_pattern`: `r'(?u)\b\w\w+\b'`
  - `max_features`: `None`
- **Step 2 (`classifier`)**: `RandomForestClassifier`
  - `n_estimators`: `100`
  - `criterion`: `'gini'`
  - `max_depth`: `None`
  - `max_features`: `'sqrt'`
  - `min_samples_split`: `2`, `min_samples_leaf`: `1`
  - `bootstrap`: `True`
  - `random_state`: `42`

### 5.2 Baseline Audit Status & Missing Evidence
- **Source Scripts**: Original training scripts, preprocessing code, and notebooks for the Random Forest and Naive Bayes baselines are **not present** within `G:\FORCORA`.
- **Naive Bayes Artifact**: No serialized Naive Bayes model file was found in local storage.
- **Input Text Discrepancy**: It remains unverified whether historical baseline training consumed the raw `Review` column or preprocessed `Processed` column.
- **Historical Metrics**: No official log files, validation curves, or test metric reports exist for the baseline models.
- **Academic Reporting Standard**: In accordance with academic integrity guidelines, baseline performance metrics cannot be presumed or fabricated. They are marked **UNVERIFIED / PENDING FORMAL EVALUATION**.

---

## 6. DistilBERT Architecture

To overcome the vocabulary limitations and lexical sparsity of bag-of-words models, the pipeline employs **DistilBERT** (`distilbert-base-uncased`), a compact transformer pre-trained via knowledge distillation from BERT:
- **Base Architecture**: 6-layer bidirectional Transformer encoder.
- **Hidden Dimension**: $d_{model} = 768$.
- **Attention Heads**: 12 multi-head self-attention mechanisms per layer.
- **Intermediate Dimension**: 3072.
- **Vocabulary Size**: 30,522 WordPiece tokens.
- **Total Parameters**: ~66.95 million (40% fewer parameters and 60% faster inference than BERT-base while retaining 97% of language understanding capabilities).
- **Classification Head**: First token (`[CLS]`) embedding extraction $\rightarrow$ Dropout ($p=0.2$) $\rightarrow$ Linear projection ($768 \rightarrow 3$).
- **Loss Function**: Multi-class Cross-Entropy Loss:
  $$\mathcal{L}_{CE} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

---

## 7. Fine-Tuning Configuration & Optimization

Training was conducted under strict reproducibility controls on local hardware using custom PyTorch training loops with Hugging Face components:

### 7.1 Optimization Hyperparameters
- **Input Column**: Raw `Review` text (preserving natural casing and syntactic punctuation).
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Weight Decay**: 0.01 (applied to weights; excluded from bias and LayerNorm parameters).
- **Peak Learning Rate**: $2 \times 10^{-5}$
- **Learning Rate Schedule**: Linear decay with 10% initial warmup:
  - Total Training Steps: 2,394 updates across 3 epochs (798 optimizer updates/epoch).
  - Warmup Steps: 239 updates.
- **Per-Device Batch Size**: 8 samples.
- **Gradient Accumulation**: 2 steps.
- **Effective Batch Size**: 16 samples.
- **Gradient Clipping**: Maximum gradient norm = 1.0.
- **Precision**: Mixed Precision (FP16 via `torch.cuda.amp.GradScaler`).
- **Sequence Processing**: Dynamic batch padding up to batch maximum, capped at `max_length = 256`.
- **Random Seed**: 42 (fixed across Python `random`, NumPy, PyTorch CPU, and CUDA RNGs).

### 7.2 Hardware Execution Details
- **Hardware**: NVIDIA GeForce RTX 2060 SUPER (8,192 MiB VRAM).
- **Software Stack**: Python 3.12 (`E:\ML\.venv`), PyTorch 2.12.1+cu130, Transformers 5.24.0.
- **Training Epoch Durations**: Epoch 1: 61.17s | Epoch 2: 56.48s | Epoch 3: 56.14s (Total: 173.79s).

---

## 8. Validation-Based Model Selection & Checkpoint Strategy

Checkpoint selection was governed strictly by **Validation Macro-F1** computed on `val_1595.csv` at the conclusion of each training epoch:

### 8.1 Training & Validation Trajectory

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation Macro-Precision | Validation Macro-Recall | Validation Macro-F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.5115 | 0.3694 | 86.96% | 0.8690 | 0.8531 | **0.8592** | Checkpoint saved |
| **2** | 0.2722 | 0.3532 | 87.84% | 0.8668 | 0.8765 | **0.8711** | Checkpoint saved |
| **3** | **0.1789** | 0.4031 | **88.65%** | **0.8759** | **0.8815** | **0.8785** | **Best Model Selected** |

### 8.2 Checkpoint Integrity
The best performing model occurred at **Epoch 3** with a Macro-F1 score of **0.8785**. Its full model weights, configuration, tokenizer vocabulary, and complete optimizer/scheduler/RNG states were preserved at `G:\FORCORA\outputs\checkpoints\best_model`.

---

## 9. Final Test Results

The frozen model from `outputs/checkpoints/best_model` was evaluated in a single inference pass on the unobserved test dataset `test_1594.csv` ($N = 1,594$):

### 9.1 Overall Metrics
- **Test Accuracy**: **88.27%** (1,407 / 1,594 correct predictions)
- **Macro-Averaged Precision**: **0.8701**
- **Macro-Averaged Recall**: **0.8732**
- **Macro-Averaged F1-Score**: **0.8716**
- **Weighted-Averaged F1-Score**: **0.8828**

### 9.2 Per-Class Performance Breakdown

| Class ID | Class Name | Support ($N$) | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Feature Request | 359 | 0.8615 | 0.8663 | **0.8639** |
| **1** | Bug Report | 420 | 0.8271 | 0.8429 | **0.8349** |
| **2** | Privacy-Irrelevant | 815 | 0.9217 | 0.9104 | **0.9160** |

### 9.3 Confusion Matrix Analysis
$$\begin{pmatrix} & \hat{y}=0 & \hat{y}=1 & \hat{y}=2 \\ y=0 & \mathbf{311} & 29 & 19 \\ y=1 & 22 & \mathbf{354} & 44 \\ y=2 & 28 & 45 & \mathbf{742} \end{pmatrix}$$

- **Class 0 (Feature Request)**: 311 true positives, 29 misclassified as Bug Reports, 19 misclassified as Irrelevant.
- **Class 1 (Bug Report)**: 354 true positives, 22 misclassified as Feature Requests, 44 misclassified as Irrelevant.
- **Class 2 (Privacy-Irrelevant)**: 742 true positives, 28 misclassified as Feature Requests, 45 misclassified as Bug Reports.

### 9.4 Operational Efficiency
- **Inference Latency**: **1.26 ms** per sample (batch size 8).
- **Total Test Inference Time**: **2.00 seconds** for all 1,594 reviews.
- **Peak GPU Memory Allocation**: **436.6 MiB** VRAM.

---

## 10. Limitations & Future Work

1. **Context Truncation Ceiling**: While sequence length 256 covers 99.78% of the dataset, long narrative reviews exceeding this boundary (0.22%) are truncated, potentially omitting trailing software requirements.
2. **Domain-Specific Ambiguity**: Inter-class confusion between Feature Requests and Bug Reports (29 FRs predicted as BR; 22 BRs predicted as FR) stems from natural linguistic overlap in user reporting (e.g., *"This app collects my location without an option to turn it off"* can be interpreted as either a defect or a request for a toggle switch).
3. **TF-IDF Baseline Provenance Gap**: Inability to locate historical baseline training code leaves baseline comparison contingent on re-evaluating the serialized pipeline or training standard baselines from scratch on identical splits.
4. **Generalization Beyond Current Corpus**: The dataset draws from mobile privacy reviews; cross-domain validation on desktop or enterprise application privacy feedback is required before generalized production deployment.
