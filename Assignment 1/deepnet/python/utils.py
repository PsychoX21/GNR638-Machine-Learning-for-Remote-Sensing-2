import random
import deepnet_backend as backend

def seed_everything(seed):
    """
    Seed all random number generators for determinism.
    """
    random.seed(seed)
    backend.manual_seed(seed)
