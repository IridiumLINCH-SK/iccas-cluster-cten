"""Metrics for CTEN models."""

import torch
import scipy.stats


def cten_exact_data_squared_error(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    CTEN Exact Data MSE metric.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    # Ground truth value and label
    y_true_val = y_true[:, 0]
    y_true_label = y_true[:, 1]
    loss = []
    for i in range(len(y_pred)):
        # Get exact data
        if y_true_label[i] == 1:
            sub = y_pred[i] - y_true_val[i]
            loss.append(sub)
    return torch.mean(torch.tensor(loss) ** 2)


def cten_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    CTEN MSE metric.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true[:, 0]
    y_true_label = y_true[:, 1]
    loss = []
    for i in range(len(y_pred)):
        sub = y_pred[i] - y_true_val[i]
        # Compute the error based on the data label
        if y_true_label[i] == 1:
            loss.append(sub)
        else:
            if sub < 0:
                loss.append(0)
            else:
                loss.append(sub)
    return torch.mean(torch.tensor(loss) ** 2)


def cten_exact_data_absolute_error(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    CTEN Exact Data MAE metric。

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true[:, 0]
    y_true_label = y_true[:, 1]
    loss = []
    for i in range(len(y_pred)):
        if y_true_label[i] == 1:
            sub = y_pred[i] - y_true_val[i]
            loss.append(sub)
    return torch.mean(torch.abs(torch.tensor(loss)))


def cten_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    CTEN MAE metric。

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true[:, 0]
    y_true_label = y_true[:, 1]
    loss = []
    for i in range(len(y_pred)):
        sub = y_pred[i] - y_true_val[i]
        if y_true_label[i] == 1:
            loss.append(sub)
        else:
            if sub < 0:
                loss.append(0)
            else:
                loss.append(sub)
    return torch.mean(torch.abs(torch.tensor(loss)))


def cten_exact_data_pearson(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    CTEN Exact Data Pearson Correlation Coefficient metric。

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true[:, 0]
    y_true_label = torch.tensor(y_true[:, 1].tolist())

    exact_data_indices = torch.where(y_true_label == 1)[0]

    return scipy.stats.pearsonr(
        y_pred[exact_data_indices].tolist(), y_true_val[exact_data_indices].tolist()
    ).statistic


CTENMETRICS = {
    "MAE": cten_absolute_error,
    "EDMAE": cten_exact_data_absolute_error,
    "MSE": cten_squared_error,
    "EDMSE": cten_exact_data_squared_error,
    "EDPearson": cten_exact_data_pearson,
}
"""Defined CTEN metrics"""
