"""CTEN model."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Union
from sklearn.preprocessing import StandardScaler
from .data import (
    CTENDataSet,
    atom_feature_standardize,
    molecule_prop_process,
    input_data_construct,
    data_set_collate_fn,
)
from ..base import *
from ...loss import CTENLoss
from ...utils.cten_metric import CTENMETRICS


class CTENModule(BaseCTENModule):
    """
    CTEN model structure.

    Parameters
    ----------
    embed_dim: int
        Dimension of the atom feature vector, must divisible by `num_heads`.
    embedding_matrix: numpy.ndarray
        Initial matrix of atom feature vector table, the first row padded with zero vector.
    num_global_features: int
        Number of global features.
    num_heads: int
        Number of parallel attention heads in each Transformer Encoder layer.
    num_transformer_layers: int
        Number of Transformer Encoder layers.
    fnn_layers: list
        Number of nodes of each FNN layer, in descending order.
        Number of FNN layers will be derived from it.
    num_output_features: int
        Number of output targets.
    transformer_dropout: float
        Dropout rate of Transformer Encoder layers.
    fnn_dropout: float
        Dropout rate of FNN layers.
    layer_norm_eps: float
        The epsilon value of layer normalization in Transformer Encoder layers.
    freeze_embedding: bool
        Keep atom feature vector table not get updated in the training process.
    dtype: torch.dtype
        Data type of the model, default to `torch.float64`.
    """

    def __init__(
        self,
        embed_dim: int,
        embedding_matrix: np.ndarray,
        num_global_features: int,
        num_heads: int,
        num_transformer_layers: int,
        fnn_layers: list[int],
        num_output_features: int,
        transformer_dropout: float,
        fnn_dropout: float,
        layer_norm_eps: float,
        freeze_embedding: bool,
        dtype: torch.dtype,
    ):
        super().__init__(
            embed_dim,
            num_global_features,
            num_heads,
            num_transformer_layers,
            fnn_layers,
            num_output_features,
            transformer_dropout,
            fnn_dropout,
            layer_norm_eps,
            dtype,
        )

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=dtype),
            freeze=freeze_embedding,
            padding_idx=0,
        )

    def forward(
        self, atom_inputs: torch.Tensor, global_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        CTEN model computation.

        Parameters
        ----------
        atom_inputs: torch.Tensor
            Lists of cluster atom elemental symbol indices.
        global_inputs: torch.Tensor
            Other global features of the cluster reaction.

        Returns
        -------
        torch.Tensor
        """
        # Generate padding mask (True for padded)
        padding_mask = atom_inputs == 0

        # Embedding layer to get atom feature vectors
        x = self.embedding(atom_inputs)

        return self.partial_forward(x, global_inputs, padding_mask)


class CTEN(BaseModel):
    """
    CTEN(Cluster Transformer Encoder Net).

    Parameters
    ----------
    atom_table: pandas.DataFrame
        Atom feature vector table to initialize CTEN embedding layer.
        Each row for a type of element, and each column for a type of atom feature.
    atom_column_name: str
        Column name of elemental symbol in the atom feature vector table.
    cluster_column_name: str
        Column name of cluster chemical formula in the input data table.
    mole_column_name: str
        Column name of chemical formula of molecule involved in the reaction in the input data table.
    mole_table: pandas.DataFrame
        Feature vector table of molecules involved in the cluster reaction.
    mole_table_column_name: str
        Column name of chemical formula in the molecule feature vector table.
    num_heads: int
        Number of parallel attention heads in each Transformer Encoder layer.
    layer_norm_eps: float
        The epsilon value of layer normalization in Transformer Encoder layers.
    num_transformer_layers: int
        Number of Transformer Encoder layers.
    fnn_layers: list
        Number of nodes of each FNN layer, in descending order.
        Number of FNN layers will be derived from it.
    transformer_dropout: float
        Dropout rate of Transformer Encoder layers.
    fnn_dropout: float
        Dropout rate of FNN layers.
    num_output_features: int, optional
        Number of output targets.
    freeze_embedding: bool, optional
        Keep atom feature vector table not get updated in the training process.
        Default to keep unchanged.
    extend_embed_vector: bool, optional
        Extend the dimension of atom feature vector to the exponent of 2,
        so that it can easily correspond to `num_heads`. Default to extend.
    loss: str, callable, optional
        Loss function. Could be a `str` corresponding to the function defined by the package,
        see `cten_loss.CTENLOSSES`. Or a custom loss function, with prediction and ground truth as inputs,
        loss value as outputs, and data type ``torch.Tensor``.
        The ground truth containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
        Default to adjusted CTEN MSE loss.
    optimizer: torch.optim.Optimizer, optional
        Optimizer of the model, see `torch.optim`. Default to `Adam`.
    device: torch.device, optional
        Device to compute the model. If not specified, device will be automatically chosen.
    dtype: torch.dtype
        Data type of the model, default to `torch.float64`.
    """

    def __init__(
        self,
        atom_table: pd.DataFrame,
        atom_column_name: str,
        cluster_column_name: str,
#        mole_column_name: str,
#        mole_table: pd.DataFrame,
#        mole_table_column_name: str,
        num_heads: int,
        layer_norm_eps: float,
        num_transformer_layers: int,
        fnn_layers: list,
        transformer_dropout: float,
        fnn_dropout: float,
        num_output_features: int = 1,
        freeze_embedding: bool = True,
        extend_embed_vector: bool = True,
        loss: Union[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = "mod_squared_loss",
        optimizer: optim.Optimizer = optim.Adam,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(CTENLoss(loss), optimizer, device=device, dtype=dtype)
        self.num_output_features = num_output_features
        self.num_heads = num_heads
        self.extend_embed_vector = extend_embed_vector
        self.layer_norm_eps = layer_norm_eps
        self.num_transformer_layers = num_transformer_layers
        self.atom_column_name = atom_column_name
        self.fnn_layers = fnn_layers
        self.transformer_dropout = transformer_dropout
        self.fnn_dropout = fnn_dropout
        self.cluster_column_name = cluster_column_name
#        self.mole_column_name = mole_column_name
#        self.mole_table = mole_table
#        self.mole_table_column_name = mole_table_column_name
        self.freeze_embedding = freeze_embedding

        # Standardize atom feature vectors
        self.atom_table = atom_feature_standardize(atom_table, atom_column_name)

        # Standard scaler of feature vector of molecules involved in the cluster reaction
#        self.scalers["mole_scaler"] = StandardScaler()
        # Standard scaler of other relevant features of the reaction
        self.scalers["other_feature_scaler"] = StandardScaler()
        # Standard scaler of number of atoms of the cluster
        self.scalers["num_atoms_scaler"] = StandardScaler()

    def process_data(
        self, X: pd.DataFrame, train_set: bool = False
    ) -> tuple[list[torch.Tensor], pd.DataFrame]:
        """
        Process input data set into cluster atom lists,
        and other global feature table of the reaction.
        And do standardization.

        Parameters
        ----------
        X: pandas.DataFrame
            Input data set.
        train_set: bool, optional
            Whether input data set is training set. Default to `False`.
            If specified to be `True`, standard scalers will be fitted.

        Returns
        -------
        tuple
        """
        # Get cluster part, molecule part, other feature part of the input
        clusters = X[[self.cluster_column_name]]
#        molecules = X[[self.mole_column_name]]
#        other_props = X.drop(columns=[self.cluster_column_name, self.mole_column_name])
        other_props = X.drop(columns=[self.cluster_column_name])

        # Transform chemical formulas of molecules involved in the reaction and standardize
#        molecules = molecule_prop_process(
#            molecules, self.mole_table, self.mole_table_column_name
#        )
#        if train_set:
#            self.scalers["mole_scaler"].fit(molecules)
#        molecules = pd.DataFrame(
#            self.scalers["mole_scaler"].transform(molecules),
#            columns=molecules.columns.tolist(),
#        )
        # Standardize other features
        if not other_props.empty:
            if train_set:
                self.scalers["other_feature_scaler"].fit(other_props)
            other_props = pd.DataFrame(
                self.scalers["other_feature_scaler"].transform(other_props),
                columns=other_props.columns.tolist(),
            )
        # Analyze cluster chemical formulas, construct and process input data tables
        clusters, cluster_num_atoms, global_features = input_data_construct(
            clusters,
            other_props,
#            molecules,
            self.atom_table,
            self.atom_column_name,
        )
        # Standardize number of atoms of the cluster
        if train_set:
            self.scalers["num_atoms_scaler"].fit(cluster_num_atoms)
        cluster_num_atoms = pd.DataFrame(
            self.scalers["num_atoms_scaler"].transform(cluster_num_atoms),
            columns=["cluster_num_atoms"],
        )
        # Concatenate global features
        global_features = pd.concat([cluster_num_atoms, global_features], axis=1)

        return clusters, global_features

    def save(
        self,
        save_dir: str,
        model_name: str,
#        mole_scaler_name: str,
        num_atoms_scaler_name: str,
        other_feature_scaler_name: str,
    ):
        """
        Save the model.

        Parameters
        ----------
        save_dir: str
            Directory to save.
        model_name: str
            File name of the model.
        mole_scaler_name: str
            File name of the standard scaler of feature vector
            of molecules involved in the cluster reaction.
        num_atoms_scaler_name: str
            File name of the standard scaler of number of atoms of the cluster.
        other_feature_scaler_name: str
            File name of the standard scaler of other relevant features of the reaction.
        """
        super().save(
            save_dir,
            model_name,
#            mole_scaler=mole_scaler_name,
            num_atoms_scaler=num_atoms_scaler_name,
            other_feature_scaler=other_feature_scaler_name,
        )

    def load(
        self,
        model: str,
#        mole_scaler: str,
        num_atoms_scaler: str,
        other_feature_scaler: str,
    ):
        """
        Load model.

        Parameters
        ----------
        model: str
            File path of the model.
        mole_scaler: str
            File path of the standard scaler of feature vector
            of molecules involved in the cluster reaction.
        num_atoms_scaler: str
            File path of the standard scaler of number of atoms of the cluster.
        other_feature_scaler: str
            File path of the standard scaler of other relevant features of the reaction.
        """
        super().load(
            model,
#            mole_scaler=mole_scaler,
            num_atoms_scaler=num_atoms_scaler,
            other_feature_scaler=other_feature_scaler,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        epochs: int,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        val_batch_size: int = 1024,
        monitor: str = "MAE",
        record_file: str = None,
    ):
        """
        Train model with validation.

        Parameters
        ----------
        X: pandas.DataFrame
            Input data of the training set.
        y: pandas.DataFrame
            Ground truth of the training set, containing two columns, the first of which is the value,
            and the second is the label of the value: 1 for exact or 0 for upper bound.
        epochs: int
            Number of epochs to train the model.
        batch_size: int, optional
            Number of samples per gradient update.
        learning_rate: float, optional
            Learning rate.
        X_val: pandas.DataFrame, optional
            Input data of the validation set. If specified both with `y_val`,
            model will be validated after trained in each epoch,
            and restored to the best model among epochs when training finishes.
        y_val: pandas.DataFrame, optional
            Ground truth of the validation set, containing two columns, the first of which is the value,
            and the second is the label of the value: 1 for exact or 0 for upper bound.
        val_batch_size: int, optional
            Number of samples per validation batch.
        monitor: str, optional
            Metric to be monitored and select the best model when validating.
            Could be `"MAE", "EDMAE", "EDMSE", "MSE"`, see `cten_metric`.
            Default to `"MAE"`.
        record_file: str, optional
            Path to save a CSV file. If specified, metrics will be recorded in each epoch.
        """
        # Process data
        clusters, global_features = self.process_data(X, True)

        # Dimension of the atom feature vector
        embedding_dim = len(self.atom_table.columns) - 1
        # Number of global features
        num_global_features = len(global_features.columns)

        # Construct initial matrix of atom feature vector table
        embedding_matrix = np.concatenate(
            (
                np.zeros((1, embedding_dim)),
                self.atom_table.drop(columns=[self.atom_column_name])
                .to_numpy()
                .tolist(),
            )
        )

        if self.extend_embed_vector:
            # Extend the dimension of atom feature vector to the exponent of 2
            nearest_2_power = 2 ** np.ceil(np.log2(embedding_dim)).astype(int).tolist()
            diff = nearest_2_power - embedding_dim
            embedding_dim = nearest_2_power
            embedding_matrix = np.concatenate(
                [embedding_matrix, np.zeros((embedding_matrix.shape[0], diff))], -1
            )

        # Set DataLoader
        train_set = CTENDataSet(clusters, global_features, y, self.dtype)

        if X_val is not None:
            # Process validation data
            clusters_val, global_features_val = self.process_data(X_val)
            val_set = CTENDataSet(clusters_val, global_features_val, y_val, self.dtype)
        else:
            val_set = None

        # Construct model
        self.module = CTENModule(
            embedding_dim,
            embedding_matrix,
            num_global_features,
            self.num_heads,
            self.num_transformer_layers,
            self.fnn_layers,
            self.num_output_features,
            self.transformer_dropout,
            self.fnn_dropout,
            self.layer_norm_eps,
            self.freeze_embedding,
            self.dtype,
        )

        # Fit the model
        super().fit(
            train_set,
            epochs,
            batch_size,
            data_set_collate_fn,
            learning_rate,
            val_set,
            val_batch_size,
            CTENMETRICS[monitor],
            CTENMETRICS,
            record_file,
        )

    def predict(self, X: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
        """
        Model predict. Returns predictions.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be predicted, containing only input features.
        batch_size: int, optional
            Number of samples per prediction batch.

        Returns
        -------
        numpy.ndarray
        """
        clusters, global_features = self.process_data(X)
        data_set = CTENDataSet(clusters, global_features, dtype=self.dtype)

        return super().predict(data_set, data_set_collate_fn, batch_size)
