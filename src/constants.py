import torch
import random
from enum import Enum

# Make cpu as torch.
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print("DEVICE: ", DEVICE)

class Task(Enum):
    # Worker performs training (returns: parameters)
    TRAIN = "train"
    # Worker performs evaluation (returns: reward)
    EVAL = "eval"
    # Worker performs data collection (returns: replay buffer)
    COLLECT = "collect"