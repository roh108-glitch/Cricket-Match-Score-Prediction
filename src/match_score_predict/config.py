from pathlib import Path

# This file is at src/match_score_predict/config.py,
# so the project root is 2 levels up from the package folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "t20_cricket_match_score_prediction.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "random_forest_model.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42