# FORCORA — Privacy Review Classification with DistilBERT

A modular PyTorch + Hugging Face Transformers pipeline for 3-class classification of app privacy reviews:

- `0`: **Feature Request**
- `1`: **Bug Report**
- `2`: **Privacy-Irrelevant**

## Project Structure

```text
FORCORA/
├── DATA/RAW/privacy_reviews/   # Train (12,756), validation (1,595), and test (1,594) CSV splits
├── src/                        # Core package (config, dataset, metrics, trainer)
├── outputs/                    # Training logs, config, documentation, and evaluation reports
├── run_train.py                # CLI training entrypoint
├── eval_test.py                # CLI final test set evaluation entrypoint
├── gui.py                      # Desktop GUI (inference, training, evaluation, results)
├── test.py                     # Dataset integrity verification script
├── pyproject.toml              # Package metadata and console entrypoints
└── requirements.txt            # Python dependencies
```

## Installation

1. Unzip the repository and enter the project directory:
   ```bash
   unzip FORCORA-repo.zip
   cd FORCORA
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```
   Or install dependencies directly:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Verify Dataset Integrity
```bash
python test.py
```

### 2. Dry-Run Pipeline Check
Validates data loading, tokenization, and a single forward pass without training:
```bash
python run_train.py --dry_run
```

### 3. Train the Model
```bash
python run_train.py
```
If installed via `pip install -e .`, you can also run:
```bash
forcora-train
```

Optional flags:
- `--epochs` (default: `3`)
- `--batch_size` (default: `8`)
- `--grad_accum` (default: `2`)
- `--lr` (default: `2e-5`)
- `--max_length` (default: `256`)
- `--seed` (default: `42`)
- `--no_fp16` (disable FP16 mixed precision)
- `--resume_from <checkpoint_dir>`

Training saves the best checkpoint (by validation macro-F1) to `outputs/checkpoints/best_model/`.

### 4. Evaluate on the Held-Out Test Set
After training produces `outputs/checkpoints/best_model/`:
```bash
python eval_test.py
```
Or, if installed via `pip install -e .`:
```bash
forcora-eval
```
### 5. Launch the Desktop GUI
```bash
python gui.py
```
Or, if installed via `pip install -e .`:
```bash
forcora-gui
```
The GUI provides four tabs:
- **Classify Review** — load a trained checkpoint (`outputs/checkpoints/best_model/`) and classify custom review text with per-class confidence bars.
- **Train & Evaluate** — run dataset checks, dry runs, model training, and test evaluation with live console output.
- **Evaluation Results** — view the test evaluation summary, training history, and confusion matrix.
- **Visualizer** — interactive charts for training/validation loss & metrics curves, per-class test metrics, and dataset class distribution.

Evaluation artifacts (metrics JSON, classification report, summary, and confusion matrix PNG) are written to `outputs/test_evaluation/`.
