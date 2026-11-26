from pathlib import Path
import pandas as pd

# Determine project root relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent

# Actual data directory using DATA/RAW structure
DATA_DIR = PROJECT_ROOT / "DATA" / "RAW" / "privacy_reviews"

# Dataset configuration with expected row counts
DATASETS = [
    {
        "filename": "original_12756 (1).csv",
        "expected_rows": 12756,
        "split": "Training",
    },
    {
        "filename": "val_1595.csv",
        "expected_rows": 1595,
        "split": "Validation",
    },
    {
        "filename": "test_1594.csv",
        "expected_rows": 1594,
        "split": "Testing",
    },
]

EXPECTED_COLUMNS = ["Review", "Processed", "Label"]

all_checks_passed = True

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

for item in DATASETS:
    file_path = DATA_DIR / item["filename"]
    print("\n" + "=" * 60)
    print(f"Split: {item['split']}")
    print(f"Filename: {item['filename']}")
    print(f"Resolved path: {file_path}")

    if not file_path.exists():
        print(f"[FAIL] File not found: {file_path}")
        all_checks_passed = False
        continue

    df = pd.read_csv(file_path)

    print(f"Shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")

    # Verification: columns
    cols_present = all(col in df.columns for col in EXPECTED_COLUMNS)
    if cols_present:
        print(f"[PASS] Expected columns present: {EXPECTED_COLUMNS}")
    else:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        print(f"[FAIL] Missing expected columns: {missing}")
        all_checks_passed = False

    # Verification: row count
    actual_rows = len(df)
    expected_rows = item["expected_rows"]
    if actual_rows == expected_rows:
        print(f"[PASS] Row count matches expected: {actual_rows:,}")
    else:
        print(f"[FAIL] Row count mismatch: expected {expected_rows:,}, got {actual_rows:,}")
        all_checks_passed = False

print("\n" + "=" * 60)
if all_checks_passed:
    print("SUMMARY: ALL CHECKS PASSED SUCCESSFULLY.")
else:
    print("SUMMARY: SOME CHECKS FAILED.")
