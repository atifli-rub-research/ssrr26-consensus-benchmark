from ultralytics import YOLO
from pathlib import Path
import random
import numpy as np
import torch


# ============================================================================
# CONFIG
# ============================================================================

WEIGHTS = "yolo26n.pt"

# YAML is inside the generated dataset folder
DATASET_YAML = "preprocessed_tiled/pfm1.yaml"

EPOCHS_LIST = [10, 20, 40, 70, 80, 110, 130, 150, 170, 200]

IMG_SIZE = 1024
BATCH_SIZE = 16
WORKERS = 0

PROJECT_DIR = "runs/train_ensemble"
BASE_RUN_NAME = "pfm1_yolo26"

BASE_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# TRAINING LOOP
# ============================================================================

Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

for i, epochs in enumerate(EPOCHS_LIST):

    seed = BASE_SEED + i

    run_name = f"{BASE_RUN_NAME}_epochs_{epochs}_seed_{seed}"

    print("=" * 80)
    print(f"Training model {i + 1}/{len(EPOCHS_LIST)}")
    print(f"Epochs : {epochs}")
    print(f"Seed   : {seed}")
    print(f"Run    : {run_name}")
    print("=" * 80)

    set_seed(seed)

    model = YOLO(WEIGHTS)

    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=run_name,
        seed=seed,
        exist_ok=False,
    )

print("\nDone training all ensemble models.")

