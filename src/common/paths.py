from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
STUDENT_TEST_DIR = DATA / "student_test"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
