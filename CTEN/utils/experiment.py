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


def scatter_plot(model: BaseModel, X: pd.DataFrame, y: pd.DataFrame):
    """
    Draw prediction scatter plot.

    Parameters
    ----------
    X: pandas.DataFrame
        Data to be predicted, containing only input features.
    y: pandas.DataFrame
        Ground truth of the data, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.

    Notes
    -----
    Only part of styles are set. Further treatments are needed to customize styles.
    """
    matplotlib.rcParams["font.family"] = "Times New Roman"
    # Get predictions
    y_pred = model.predict(X)
    # Compute data range
    min_val = min(np.min(y_pred), np.min(y.iloc[:, 0])) - 1
    max_val = max(np.max(y_pred), np.max(y.iloc[:, 0])) + 1
    # Set canvas
    plt.figure(dpi=60, figsize=(20, 20))
    # Classify by exact and upper bound data
    ext_x = []
    ext_y = []
    upp_x = []
    upp_y = []
    for i in range(len(y_pred)):
        if y.to_numpy()[i][1] == 1:
            ext_x.append(y.to_numpy()[i][0])
            ext_y.append(y_pred[i])
        else:
            upp_x.append(y.to_numpy()[i][0])
            upp_y.append(y_pred[i])
    # Draw scatter plot
    plt.scatter(
        upp_x, upp_y, marker="^", c="blue", s=300, alpha=0.7, label="upper bound"
    )
    plt.scatter(
        ext_x, ext_y, marker=".", c="red", s=1000, alpha=0.7, label="exact data"
    )
    # Draw diagonal
    plt.plot([min_val, max_val], [min_val, max_val])

    # Set styles
    plt.xticks(size=45)
    plt.yticks(size=45)
    plt.xlabel("Experimental lgk1", size=45)
    plt.ylabel("Predicted lgk1", size=45)
    plt.legend(fontsize=25)
