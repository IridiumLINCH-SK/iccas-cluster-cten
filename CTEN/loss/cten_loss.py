"""CTEN loss function."""

import torch
from torch.overrides import handle_torch_function, has_torch_function_variadic
from typing import Callable, Union
from .base import BaseLoss


class CTENLoss(BaseLoss):
    """
    CTEN loss function.

    Parameters
    ----------
    func: str, callable
        CTEN loss function. Could be a `str` corresponding to the function defined by the package,
        see :const:`CTENLOSSES`. Or a custom loss function, with prediction and ground truth as inputs,
        loss value as outputs, and data type ``torch.Tensor``.
        The ground truth containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    """

    def __init__(
        self, func: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ) -> None:
        if isinstance(func, str):
            try:
                super().__init__(CTENLOSSES[func])
            except:
                raise Exception(f"Unsupported CTEN loss function: {func}.")
        else:
            super().__init__(func)


def cten_squared_loss_mod(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CTEN MSE loss function, loss of upper bound data adjusted according to the experiment.

    Parameters
    ----------
    input: torch.Tensor
        Prediction.
    target: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.

    Returns
    -------
    torch.Tensor
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            cten_squared_loss_mod, (input, target), input, target
        )

    if not (target.size()[0] == input.size()[0]):
        raise Exception(
            "Length of the ground truth is different from that of the prediction."
        )

    # Prediction
    y_pred = input
    # Ground truth with value and label
    y_true = target
    # Ground truth value
    y_true_val = y_true[:, [0]]
    # Ground truth label
    y_true_label = y_true[:, [1]]
    # Loss of exact data
    y_loss = y_pred - y_true_val
    # Loss of upper bound data, reduced when the prediction is less than the ground truth,
    # and increased when the prediction is greater than the ground truth.
    y_loss_bound = torch.where(
        y_loss > 0, torch.square(y_loss) * 3, torch.square(y_loss) / 3
    )
    # Choose loss value according to the label
    y_loss_mod = torch.where(y_true_label == 1, torch.square(y_loss), y_loss_bound)

    return torch.mean(y_loss_mod)


def cten_squared_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CTEN MSE loss function, with upper bound data considered.

    Parameters
    ----------
    input: torch.Tensor
        Prediction.
    target: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.

    Returns
    -------
    torch.Tensor
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(cten_squared_loss, (input, target), input, target)

    if not (target.size()[0] == input.size()[0]):
        raise Exception(
            "Length of the ground truth is different from that of the prediction."
        )

    # Prediction
    y_pred = input
    # Ground truth with value and label
    y_true = target
    # Ground truth value
    y_true_val = y_true[:, [0]]
    # Ground truth label
    y_true_label = y_true[:, [1]]
    # Loss of exact data
    y_loss = y_pred - y_true_val
    # Loss of upper bound data, reduced when the prediction is less than the ground truth,
    # and increased when the prediction is greater than the ground truth.
    y_loss_bound = torch.where(y_loss > 0, torch.square(y_loss), 0)
    # Choose loss value according to the label
    y_loss_mod = torch.where(y_true_label == 1, torch.square(y_loss), y_loss_bound)

    return torch.mean(y_loss_mod)


def cten_absolute_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CTEN MAE loss function, with upper bound data considered.

    Parameters
    ----------
    input: torch.Tensor
        Prediction.
    target: torch.Tensor
        Ground truth, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.

    Returns
    -------
    torch.Tensor
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(cten_absolute_loss, (input, target), input, target)

    if not (target.size()[0] == input.size()[0]):
        raise Exception(
            "Length of the ground truth is different from that of the prediction."
        )

    # Prediction
    y_pred = input
    # Ground truth with value and label
    y_true = target
    # Ground truth value
    y_true_val = y_true[:, [0]]
    # Ground truth label
    y_true_label = y_true[:, [1]]
    # Loss of exact data
    y_loss = y_pred - y_true_val
    # Loss of upper bound data, reduced when the prediction is less than the ground truth,
    # and increased when the prediction is greater than the ground truth.
    y_loss_bound = torch.where(y_loss > 0, y_loss, 0)
    # Choose loss value according to the label
    y_loss_mod = torch.where(y_true_label == 1, y_loss, y_loss_bound)

    return torch.mean(torch.abs(y_loss_mod))


CTENLOSSES = {
    "mod_squared_loss": cten_squared_loss_mod,
    "squared_loss": cten_squared_loss,
    "absolute_loss": cten_absolute_loss,
}
"""Loss functions defined"""
