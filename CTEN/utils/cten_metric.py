"""Metrics for CTEN models."""

import torch
import scipy.stats


def mean_squared_error(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    CTEN MSE metric.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    # Ground truth value and label
    y_true_val = y_true.ravel()
    loss = []
    for i in range(len(y_pred)):
        # Get exact data
        sub = y_pred[i] - y_true_val[i]
        loss.append(sub)
    return torch.mean(torch.tensor(loss) ** 2)


def mean_absolute_error(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    CTEN MAE metric.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true.ravel()
    loss = []
    for i in range(len(y_pred)):
        sub = y_pred[i] - y_true_val[i]
        loss.append(sub)
    return torch.mean(torch.abs(torch.tensor(loss)))



def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    CTEN Pearson Correlation Coefficient metric.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth.
    y_pred: torch.Tensor
        Prediction.

    Returns
    -------
    torch.Tensor
    """
    y_pred = y_pred.ravel()
    y_true_val = y_true.ravel()


    return scipy.stats.pearsonr(
        y_pred.tolist(), y_true_val.tolist()
    ).statistic


CTENMETRICS = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "Pearson r": pearson_r,
}
"""Defined CTEN metrics"""

if __name__ == '__main__':
    t1 = torch.rand(2, 3)
    t2 = torch.rand(2, 3)
    print(CTENMETRICS["MAE"](t1, t2))
    print(CTENMETRICS["MSE"](t1, t2))
    print(CTENMETRICS["Pearson r"](t1, t2))
