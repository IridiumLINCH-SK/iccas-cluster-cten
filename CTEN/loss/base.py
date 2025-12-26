"""Base loss function class."""

import torch
import torch.nn as nn
from typing import Callable


class BaseLoss(nn.Module):
    """
    Loss function.

    Parameters
    ----------
    func: callable
        Loss function, with prediction and ground truth as inputs,
        outputs loss value, with data type ``torch.Tensor``.
    """

    def __init__(
        self, func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        if callable(func):
            self.func = func
        else:
            raise Exception(f"Unsupported loss function type: {type(func).__name__}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value.

        Parameters
        ----------
        input: torch.Tensor
            Prediction.

        target: torch.Tensor
            Ground truth.

        Returns
        -------
        torch.Tensor
        """
        return self.func(input, target)
