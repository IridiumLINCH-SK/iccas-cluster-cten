"""Data process class and functions for CTEN."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import sklearn.preprocessing
from collections import Counter
from ...utils.data import analyze_chemical_formula_table


class CTENDataSet(Dataset):
    """
    CTEN Dataset class.

    Parameters
    ----------
    atom_inputs: pandas.DataFrame
        Cluster atom lists.
    global_inputs: pandas.DataFrame
        Other global features of the cluster reaction.
    rates: pandas.DataFrame, optional
        Cluster reaction rate, containing two columns, the first of which is the value,
        and the second is the label of the value: 1 for exact or 0 for upper bound.
    dtype: torch.dtype, optional
        Data type of the model.
    """

    def __init__(
        self,
        atom_inputs: pd.DataFrame,
        global_inputs: pd.DataFrame,
        rates: pd.DataFrame = None,
        dtype: torch.dtype = torch.float64,
    ):
        # Initialize data
        self.atom_inputs = atom_inputs
        self.global_inputs = global_inputs.to_numpy()
        self.rates = rates
        self.dtype = dtype

    def __len__(self):
        # Return size of the data set
        return len(self.atom_inputs)

    def __getitem__(self, idx):
        # Return data by index
        if self.rates is None:
            return self.atom_inputs[idx], torch.tensor(
                self.global_inputs[idx], dtype=self.dtype
            )
        return (
            self.atom_inputs[idx],
            torch.tensor(self.global_inputs[idx], dtype=self.dtype),
            torch.tensor(self.rates.to_numpy()[idx], dtype=self.dtype),
        )


def data_set_collate_fn(batch: list):
    """
    Merge a list of samples to form a mini-batch of Tensors and pad atom lists.
    Used when using batched loading from a map-style dataset.
    """
    # Get inputs (and rates)
    data = list(zip(*batch))
    if len(data) == 3:
        atom_inputs, global_inputs, rates = data
    else:
        atom_inputs, global_inputs = data
        rates = None
    # Pad atom lists
    atom_inputs = nn.utils.rnn.pad_sequence(atom_inputs, batch_first=True)

    if rates is None:
        return atom_inputs, torch.stack(global_inputs)

    return atom_inputs, torch.stack(global_inputs), torch.stack(rates)


def molecule_prop_process(
    molecule: pd.DataFrame, mole_table: pd.DataFrame, mole_table_column_name: str
) -> pd.DataFrame:
    """
    Transform chemical formulas of molecules involved in the reaction into feature vectors.

    Parameters
    ----------
    molecule: pandas.DataFrame
        List of chemical formulas of molecules involved in the reaction.
    mole_table: pandas.DataFrame
        Feature vector table of molecules.
    mole_table_column_name: str
        Column name of chemical formula in the feature vector table.

    Returns
    -------
    pandas.DataFrame
    """
    mole_features = []
    # Generate vector for each chemical formula input
    for i in range(len(molecule)):
        mole_features.append(
            mole_table[mole_table[mole_table_column_name] == molecule.iloc[i, 0]]
            .to_numpy()
            .tolist()[0]
        )
    # Delete chemical formula column and construct DataFrame
    mole_features = pd.DataFrame(
        mole_features, columns=mole_table.columns.tolist()
    ).drop(columns=[mole_table_column_name])
    return mole_features


def input_data_construct(
    clusters: pd.DataFrame,
    other_props: pd.DataFrame,
#    mole_props: pd.DataFrame,
    atom_props: pd.DataFrame,
    atom_column_name: str,
) -> tuple[list[torch.Tensor], pd.DataFrame, pd.DataFrame]:
    """
    Construct cluster atom lists and cluster atom number table,
    and process other global feature table.

    Parameters
    ----------
    clusters: pandas.DataFrame
        List of cluster chemical formulas.
    other_props: pandas.DataFrame
        Table of other relevant features of the reaction.
    mole_props: pandas.DataFrame
        List of feature vectors of molecules involved in the reaction.
    atom_props: pandas.DataFrame
        Atom feature vector table.
    atom_column_name: str
        Column name of elemental symbol in the atom feature vector table.

    Returns
    -------
    tuple
    """
    # Analyze chemical formulas
    atom_charge_counts, _ = analyze_chemical_formula_table(clusters)
    cluster_matrix = []
    cluster_num_atoms = []
    global_features = []
    # Generate inputs for each chemical formula
    for i in range(len(atom_charge_counts)):
        # Get atom lists
        atom_counter = Counter(
            {
                atom: count
                for atom, count in atom_charge_counts[i].items()
                if atom != "charge"
            }
        )
        x_row_data = []
        # Transform elemental symbol into row number of atom feature vector table corresponding to it
        for atom in list(atom_counter.elements()):
            x_row_data.append(
                atom_props[atom_props[atom_column_name] == atom].index[0] + 1
            )
        cluster_matrix.append(torch.tensor(x_row_data))
        # Record number of atoms of the cluster
        cluster_num_atoms.append([len(x_row_data)])
        # Record other global features
        global_features.append(
            [atom_charge_counts[i]["charge"]]
            + other_props.iloc[i].to_numpy().tolist()
#            + mole_props.iloc[i].to_numpy().tolist()
        )
    return (
        cluster_matrix,
        pd.DataFrame(cluster_num_atoms, columns=["cluster_num_atoms"]),
        pd.DataFrame(
            global_features,
            columns=["charge"]
            + other_props.columns.tolist()
#            + mole_props.columns.tolist(),
        ),
    )


def atom_feature_standardize(
    atom_props: pd.DataFrame, atom_column_name: str
) -> pd.DataFrame:
    """
    Standardize atom feature vector table.

    Parameters
    ----------
    atom_props: pandas.DataFrame
        Atom feature vector table.
    atom_column_name: str
        Column name of elemental symbol in the atom feature vector table.

    Returns
    -------
    pandas.DataFrame
    """
    std = sklearn.preprocessing.StandardScaler().fit_transform(
        atom_props.drop([atom_column_name], axis=1)
    )
    std_data = np.column_stack([atom_props[atom_column_name].to_numpy(), std])
    prop_list = atom_props.columns.to_list()
    prop_list.remove(atom_column_name)
    return pd.DataFrame(std_data, columns=[atom_column_name] + prop_list)
