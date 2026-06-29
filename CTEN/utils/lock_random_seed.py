"""Tools for conducting experiments."""

import random
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ..model import BaseModel


def set_random_seed(seed: int):
    """
    Set random seed.

    Parameters
    ----------
    seed: int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


