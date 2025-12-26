"""Loss functions for training models."""

from .base import BaseLoss
from .cten_loss import CTENLoss

__all__ = ["BaseLoss", "CTENLoss"]
