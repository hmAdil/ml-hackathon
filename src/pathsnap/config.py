from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
STUDENT_TEST_DIR = DATA / "student_test"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
RANDOM_SEED = 42

# candidates table contract: one row per (trip_id, obs_idx, candidate_arc_id)
KEY_COLS = ["trip_id", "obs_idx", "candidate_arc_id"]
LABEL_COL = "is_true"
