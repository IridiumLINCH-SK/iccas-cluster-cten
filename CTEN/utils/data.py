"""Tools for processing data."""

import os
import numpy as np
import pandas as pd
from typing import Union


def analyze_chemical_formula_list(formulas: list[str]) -> tuple[list[dict], list]:
    """
    Get atoms and charge of each chemical formula in the cluster list.
    Return analysis dictionary of each chemical formula and total element types of the data set.

    Parameters
    ----------
    formulas: list
        List of cluster chemical formulas.

    Returns
    -------
    tuple
    """
    results = []
    unique_atoms = set()
    # Go through each chemical formula in the list
    for formula in formulas:
        atom_counts = {}
        charge = 0
        atom = ""
        count = ""
        # For each character in the chemical formula
        for char in formula:
            # Capital letter marks the beginning of the elemental symbol
            if char.isupper():
                # `atom` has content marks the end of last elemental symbol
                # Record the element and number of the atom
                if atom:
                    atom_counts[atom] = int(count) if count else 1
                    unique_atoms.add(atom)
                # Add charactor to `atom`
                atom = char
                # Clear number of atoms
                count = ""
            # Append lower case letter to elemental symbol
            elif char.islower():
                atom += char
            # Digit represents number of atoms
            elif char.isdigit():
                count += char
            # Charge
            elif char == "-":
                charge = -1
            elif char == "+":
                charge = 1

        # Record the last element and number of atoms
        atom_counts[atom] = int(count) if count else 1
        unique_atoms.add(atom)
        # Record charges
        atom_counts["charge"] = charge
        results.append(atom_counts)

    return results, list(unique_atoms)


def analyze_chemical_formula_table(
    formulas: Union[pd.Series, pd.DataFrame],
) -> tuple[list[dict], list]:
    """
    Get atoms and charge of each chemical formula in the `pandas` tabel.
    Return analysis dictionary of each chemical formula and total element types of the data set.

    Parameters
    ----------
    formulas: pandas.Series, pandas.DataFrame
        Cluster chemical formula table.

    Returns
    -------
    tuple
    """
    if isinstance(formulas, pd.DataFrame):
        # Deal with pandas.DataFrame
        return analyze_chemical_formula_list(formulas.iloc[:, 0].tolist())
    elif isinstance(formulas, pd.Series):
        # Deal with pandas.Series
        return analyze_chemical_formula_list(formulas.tolist())
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")


def cluster_atom_distribution(
    data: pd.DataFrame,
    cluster_column_name: str,
    exact_label_name: str,
    atom_props: pd.DataFrame,
    atom_column_name: str,
) -> dict:
    """
    Count the distribution of atoms of clusters.

    Parameters
    ----------
    data: pandas.DataFrame
        Data set.
    cluster_column_name: str
        Column name of cluster chemical formula in the data set.
    exact_label_name: str
        Column name of data label(exact or upper bound) in the data set.
    atom_props: pandas.DataFrame
        Atom feature vector table.
    atom_column_name: str
        Column name of elemental symbol in the atom feature vector table.

    Returns
    -------
    dict
    """
    clusters = data[[cluster_column_name]]
    exact_label = data[[exact_label_name]]
    atoms = atom_props[[atom_column_name]].to_numpy().ravel().tolist()
    count = dict.fromkeys(atoms, 0)
    count["exact"] = 0
    count["bound"] = 0
    count["pos_charge"] = 0
    count["neg_charge"] = 0
    count["neutral"] = 0
    # Count element and charge
    for c in analyze_chemical_formula_table(clusters)[0]:
        for a in c.keys():
            if a != "charge":
                count[a] += 1
        if c["charge"] == 1:
            count["pos_charge"] += 1
        elif c["charge"] == -1:
            count["neg_charge"] += 1
        else:
            count["neutral"] += 1
    # Count exact data and upper bound data
    for l in exact_label.to_numpy().ravel():
        if l == 1:
            count["exact"] += 1
        else:
            count["bound"] += 1
    return count


def rate_distribution(data: pd.DataFrame, column_name: str) -> dict:
    """
    Count the distribution of reaction rate.

    Parameters
    ----------
    data: pandas.DataFrame
        Data set.
    column_name: str
        Column name of reaction rate in the data set.

    Returns
    -------
    dict
    """
    yrange = data[[column_name]].astype("int")
    return yrange.value_counts().to_dict()


def data_split(
    data: str,
    cluster_column_name: str,
    save_dir: str = ".",
    seed: Union[int, float, bool] = None,
):
    """
    Split data set with training : validation : testing = 8 : 1 : 1.

    Parameters
    ----------
    data: str
        Path of CSV file of the data set.
    cluster_column_name: str
        Column name of cluster chemical formula in the data set.
    save_dir: str, optional
        Directory to save the data set. Default to the current directory.
    seed: int, float, bool, optional
        Random seed. Default to not set.
    """
    # Read CSV file
    df = pd.read_csv(data)

    # Group by cluster chemical formula
    grouped = df.groupby(cluster_column_name)

    # Put groups into a list
    groups = list(grouped)

    # Shuffle the groups
    np.random.seed(seed)
    np.random.shuffle(groups)

    # Compute group numbers with proportion = 1 : 1 : 8
    num_groups = len(groups)
    split_1 = num_groups // 10  # 1/10
    split_2 = split_1  # 1/10

    # Put groups into DataFrames in proportion
    test_set = pd.concat([grouped.get_group(groups[i][0]) for i in range(split_1)])
    val_set = pd.concat(
        [grouped.get_group(groups[i][0]) for i in range(split_1, split_1 + split_2)]
    )
    train_set = pd.concat(
        [grouped.get_group(groups[i][0]) for i in range(split_1 + split_2, num_groups)]
    )

    # Save CSV files
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_set.to_csv(f"{save_dir}/test_set.csv", index=False)
    val_set.to_csv(f"{save_dir}/val_set.csv", index=False)
    train_set.to_csv(f"{save_dir}/train_set.csv", index=False)
