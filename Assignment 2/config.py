import os
import sys
import random
import numpy as np
import torch

# Seeds
SEED = int(os.environ.get("SEED", 42))

def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "train_data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
SPLIT_DIR   = os.path.join(RESULTS_DIR, "_split")
TRAIN_DIR   = os.path.join(SPLIT_DIR, "train")
VAL_DIR     = os.path.join(SPLIT_DIR, "val")

# Image
IMG_SIZE       = 224   # default for all models except inception_v3
INCEPTION_SIZE = 299   # inception_v3 expects 299x299
NUM_CLASSES    = 30

# Training
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 32))
LR_LINEAR_PROBE = 1e-3
LR_FINETUNE     = 1e-4
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS_FULL = 30
MAX_EPOCHS_FEW  = 20

NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", 0 if sys.platform == "win32" else 4))

# Models
ALL_MODELS     = ["resnet50", "inception_v3", "densenet121", "efficientnet_b0", "convnext_tiny"]
DEFAULT_MODELS = ["convnext_tiny", "inception_v3", "densenet121"]

# Scenarios
FEWSHOT_FRACS = [1.0, 0.20, 0.05]
GAUSS_SIGMAS  = [0.05, 0.1, 0.2]

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
