"""Base module for CTEN."""

import abc
import torch
import torch.nn as nn


class GlobalAveragePooling1D(nn.Module):
    """Global average pooling layer."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute global average pooling.

        Parameters
        ----------
        input: torch.Tensor
            Feature vector input lists.
        mask: torch.Tensor
            Padding mask of feature vector lists.

        Returns
        -------
        torch.Tensor
        """
        # Make dimension of mask correspond to that of input
        mask = (~mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        x = input * mask  # Make masked vectors zero

        # Average pooling on the dimension of seq_len
        return torch.sum(x, dim=1) / mask.sum(dim=1)


class BaseCTENModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Base module for CTEN.

    Parameters
    ----------
    embed_dim: int
        Dimension of the row vector of the cluster feature matrix, must divisible by `num_heads`.
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
    dtype: torch.dtype
        Data type of the model, default to `torch.float64`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_global_features: int,
        num_heads: int,
        num_transformer_layers: int,
        fnn_layers: list[int],
        num_output_features: int,
        transformer_dropout: float,
        fnn_dropout: float,
        layer_norm_eps: float,
        dtype: torch.dtype,
    ):
        super().__init__()

        # Transformer Encoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=transformer_dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            dtype=dtype,
        )
        self.transformer_layers = nn.TransformerEncoder(
            transformer_layer, num_transformer_layers, enable_nested_tensor=False
        )

        # Global average pooling layer
        self.global_avg_pool = GlobalAveragePooling1D()

        # FNN layer
        self.fnn_layers = nn.ModuleList()
        # Dimension of vectors after concatenating
        # outputs of global average pooling layer and global features
        linear_input_dim = embed_dim + num_global_features
        for units in fnn_layers:
            self.fnn_layers.append(nn.Linear(linear_input_dim, units, dtype=dtype))
            self.fnn_layers.append(nn.PReLU(units, 0, dtype=dtype))
            self.fnn_layers.append(nn.Dropout(fnn_dropout))
            linear_input_dim = units  # Update input dimension of next layer

        # output layer
        self.output_layer = nn.Linear(
            linear_input_dim, num_output_features, dtype=dtype
        )

    def partial_forward(
        self,
        cluster_feature: torch.Tensor,
        other_feature: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Partial computation of the model.

        Parameters
        ----------
        cluster_feature: torch.Tensor
            Cluster feature matrix.
        other_feature: torch.Tensor
            Vector of other features of the reaction.
        padding_mask: torch.Tensor
            Padding mask of input cluster feature matrix.

        Returns
        -------
        torch.Tensor
        """
        # Transformer Encoder layer
        x = self.transformer_layers(cluster_feature, src_key_padding_mask=padding_mask)

        # Global average pooling layer
        x = self.global_avg_pool(x, padding_mask)

        # Concatenate outputs of global average pooling layer and global features
        combined_features = torch.cat([x, other_feature], dim=-1)

        # FNN layer
        for layer in self.fnn_layers:
            combined_features = layer(combined_features)

        # output layer
        return self.output_layer(combined_features)

    @abc.abstractmethod
    def forward(self, *args):
        """Do pretreatment and then return partial_forward."""
