"""Base Model class."""

import os
import copy
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Callable
from ...loss import BaseLoss


class BaseModel:
    """
    Base Model class.

    Parameters
    ----------
    loss: BaseLoss
        Loss function object.
    optimizer: torch.optim.Optimizer
        Pytorch optimizer object.
    module: torch.nn.Module, optional
        Module inheriting from ``torch.nn.Module`` and defining model layers.
    device: torch.device, optional
        Device to compute the model. If not specified, device will be automatically chosen.
    dtype: torch.dtype, optional
        Data type of the model.
    """

    def __init__(
        self,
        loss: BaseLoss,
        optimizer: optim.Optimizer,
        module: nn.Module = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.loss = loss
        self.optimizer = optimizer
        self.dtype = dtype

        # Add scalers here
        self.scalers = {}

        self.module = module

    def save(self, save_dir: str, module_name: str, **scaler_names):
        """
        Save module and scalers of the model.

        Parameters
        ----------
        save_dir: str
            Directory to save.
        module_name: str
            File name of the module.
        scaler_names: keyword args
            File names of scalers. Key: scaler name in ``self.scalers``.
            Value: file name.
        """
        if self.module is None:
            raise Exception("Model has not been loaded or trained.")

        # mkdir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save module
        torch.save(self.module, f"{save_dir}/{module_name}.pth")

        # Save scalers
        for key, scaler_name in scaler_names.items():
            joblib.dump(self.scalers[key], f"{save_dir}/{scaler_name}.save")

    def load(self, module: str, **scalers):
        """
        Load module and scalers of the model.

        Parameters
        ----------
        module: str
            File path of module.
        scalers: keyword args
            File paths of scalers. Key: scaler name in ``self.scalers``.
            Value: file path.
        """
        self.module = torch.load(module, map_location=self.device, weights_only=False)

        for key, scaler in scalers.items():
            self.scalers[key] = joblib.load(scaler)

    def fit(
        self,
        data_set: Dataset,
        epochs: int,
        batch_size: int,
        collate_fn: Callable,
        learning_rate: float,
        val_data_set: Dataset = None,
        val_batch_size: int = None,
        monitor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        metric_funcs: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        record_file: str = None,
    ):
        """
        Train model with validation.

        Parameters
        ----------
        data_set: torch.utils.data.Dataset
            Training data set, containing input features and labels.
        epochs: int
            Number of epochs to train the model.
        batch_size: int
            Number of samples per gradient update.
        collate_fn: callable
            Function for merging a list of samples to form a mini-batch of Tensor(s).
            Used to construct a DataLoader when using batched loading from a Dataset.
        learning_rate: float
            Learning rate.
        val_data_set: torch.utils.data.Dataset, optional
            Validation data set. If specified, model will be validated after trained in each epoch,
            and restored to the best model among epochs when training finishes.
        val_batch_size: int, optional
            Number of samples per validation batch.
        monitor: callable, optional
            Function to compute the metric to be monitored and select the best model when validating.
        metric_funcs: dict, optional
            Functions to compute metrics when validating. Key: metric name.
            Value: function to compute the metric.
        record_file: str, optional
            Path to save a CSV file. If specified, metrics computed from
            metric_funcs will be recorded in each epoch.
        """
        self.module.to(self.device)
        # Decide whether validate while training
        validation = val_data_set is not None

        # set DataLoader
        train_loader = DataLoader(
            data_set, batch_size, shuffle=True, collate_fn=collate_fn
        )

        criterion = self.loss
        optimizer = self.optimizer(self.module.parameters(), lr=learning_rate)

        if validation:
            # set validation DataLoader
            val_loader = DataLoader(val_data_set, val_batch_size, collate_fn=collate_fn)

            # Information to record
            best = torch.inf
            best_weights = None
            if record_file is not None:
                metrics_record = []

        # Train
        for epoch in range(epochs):
            self.module.train()
            total_loss = 0

            # Timing
            start_time = time.time()

            for batch_data in train_loader:
                inputs = [
                    data.to(self.device) for data in batch_data[: len(batch_data) - 1]
                ]
                ground_truth = batch_data[-1]
                optimizer.zero_grad()
                outputs = self.module(*inputs)
                loss = criterion(outputs, ground_truth.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            end_time = time.time()

            print(
                f"Epoch: {epoch + 1}, loss: {(total_loss / len(train_loader)):.4f}, {round(end_time - start_time)}s",
                flush=True,
            )

            if validation:
                # Validate
                self.module.eval()
                predictions = []
                ground_truth = []

                start_time = time.time()

                with torch.no_grad():  # Inferring
                    for batch_data in val_loader:
                        inputs = [
                            data.to(self.device)
                            for data in batch_data[: len(batch_data) - 1]
                        ]
                        ground_truth.append(batch_data[-1].to(self.device))
                        preds = self.module(*inputs)
                        predictions.append(preds)

                # Concatenate batch results
                predictions = torch.cat(predictions)
                ground_truth = torch.cat(ground_truth)

                # Calculate metrics
                record_info = {"epoch": epoch}
                metric_info = ""
                if metric_funcs is not None:
                    for metric, func in metric_funcs.items():
                        res = func(ground_truth, predictions)
                        metric_info += f"{metric}: {res:.4f}, "
                        record_info[metric] = res.tolist()

                end_time = time.time()

                # Print validation metric
                current = monitor(ground_truth, predictions)
                print(
                    f"Validation: {metric_info}{round(end_time - start_time)}s",
                    flush=True,
                )

                # Record metrics
                if record_file is not None:
                    metrics_record.append(record_info)

                # Store best weights
                if current is not None:
                    if best_weights is None:
                        # Restore the weights after first epoch if no progress is ever made.
                        best_weights = copy.deepcopy(self.module.state_dict())

                    if current < best:
                        best = current
                        best_weights = copy.deepcopy(self.module.state_dict())

        if validation:
            # Restore best weights and save metrics
            self.module.load_state_dict(best_weights)
            if record_file is not None:
                pd.DataFrame(metrics_record).to_csv(record_file, index=False)

    def predict(
        self, data_set: Dataset, collate_fn: Callable, batch_size: int
    ) -> np.ndarray:
        """
        Model predict. Returns predictions.

        Parameters
        ----------
        data_set: torch.utils.data.Dataset
            Data set to be predicted, containing only input features.
        collate_fn: callable
            Function for merging a list of samples to form a mini-batch of Tensor(s).
            Used to construct a DataLoader when using batched loading from a Dataset.
        batch_size: int
            Number of samples per prediction batch.

        Returns
        -------
        numpy.ndarray
        """
        if self.module is None:
            raise Exception("Model has not been loaded or trained.")

        data_loader = DataLoader(data_set, batch_size, collate_fn=collate_fn)

        self.module.eval()

        predictions = []

        with torch.no_grad():  # Inferring
            for batch_data in data_loader:
                inputs = [data.to(self.device) for data in batch_data]
                preds = self.module(*inputs)
                predictions.append(preds)

        # Concatenate all batch results
        predictions = torch.cat(predictions)

        return np.array(predictions.tolist())
