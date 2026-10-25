from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

DATASET_ID = "fancyzhx/amazon_polarity"
DATASET_TOTAL_ROWS = 4_000_000
DATASET_TRAIN_ROWS = 3_600_000
DATASET_TEST_ROWS = 400_000
LABEL_NAMES = {0: "negative", 1: "positive"}
RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 10_000
