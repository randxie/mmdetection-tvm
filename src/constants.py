import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
DEPLOY_WEIGHT_DIR = os.path.join(ROOT_DIR, "deploy_weight")
TUNING_LOG_DIR = os.path.join(ROOT_DIR, "tuning_log")
