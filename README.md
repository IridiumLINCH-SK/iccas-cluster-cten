# CTEN (Cluster Transformer Encoder Network)
This is a repository of CTEN, a neural-network-based machine learning model incorporating a transformer encoder aiming to modelling quantitative properties of atomic clusters, e.g. atomization energies (AE), spin multiplicities (SM), and so on. This model is internally capable in distinguishing atomic clusters with different compositions and charges, thus a model can be trained from provided training labels WITHOUT any other features. Utilization of other features are also supported.

In the CTEN model, a cluster is decomposed into **a series of atom symbols** and its **charge**, then each atom is embedded into a vector, forming a series of "_atomic vectors_". These vectors go through _transformer encoder layers_ with _multi-head attention mechanism_, and undergo global average pooling into a vector of fixed length. The **number of atoms** in the cluster and its charge are then joined into the vector. Finally, this concatenated vector is processed by a _fully-connected neural network_ to obtain the model predicted value. 

The following authors contributed to the codes:
- [Zi-Yue Wang](https://github.com/wangziyue00/): Raised the idea of utilization of a transformer encoder and implemented it with python codes. 
- [Ning-Zheng Li (Iridium LINCH-SK)](https://github.com/IridiumLINCH-SK/): Conducted necessary modifications to the codes.

## System Requirements
The CTEN model has following requirements: 
- Python (>= 3.9)
- NumPy
- pandas
- scikit-learn
- SciPy
- PyTorch
- matplotlib

## Installation
### Installation from wheel
To install `iccas-cluster-cten` from wheel, you can download `iccas-cluster-cten` wheel and execute:

```sh
pip install iccas_cluster_cten-1.0.0-py3-none-any.whl
```

### Installation from sources

To install `iccas-cluster-cten` from source, you can clone the git repository to get source codes:

```sh
git clone https://github.com/IridiumLINCH-SK/iccas-cluster-cten
```

In the root directory of `iccas-cluster-cten` project, execute:

```sh
pip install .
```

## CTEN Quick Start
### CTEN_examples
This is an example of utilizing CTEN, corresponding to the terminating point of the learning curve in $N_\mathrm{M} = 4$ subspace. The following data points will participate in training the model:
- 12 $N_\mathrm{M} = 0$ samples
- 450 $N_\mathrm{M} = 1$ samples
- 20,822 $N_\mathrm{M} = 2$ samples
- 57,638 $N_\mathrm{M} = 3$ samples
- 9,752 $N_\mathrm{M} = 4$ samples

These samples are splitted into a 79,821-size training set and a 8,853-size validation set to alleviate the overfitting issues: During the training process, the mean absolute error (MAE) of the validation set is continuously monitored, and the model parameters are reverted to those who obtain the minimal validation MAE. The performance of this model is then tested on a 1,500-size testing set.

The folder contains:
- `atomprop.csv`: Atomic vectors at the orbital fineness.
- `train_set.csv`: Training set.
- `val_set.csv`: Validation set.
- `test_set.csv`: Testing set.
- `train_val_model.py`: The script for training a model with validation set monitoring. 
- `test_model.py`: The script for predicting target values of training and testing sets.
- `MAE_r.py`: The script for evaluating model performances. MAE and Pearson's $r$ are provided.

The typical running time of training a model with the given trainning set is about 13 s/epoch (Tested on an NVIDIA GeForce RTX 4090 GPU).

### Example_outputs
The following items will be output after running `train_val_model.py`:
- `loss_change.csv`: Recording the variation of MAE, MSE and Pearson's $r$ on the validation set during training epochs.
- `models/cten.pth`: Weights and biases of the neural network.
- `models/num_atoms_scaler.save`: The standardizer of atom numbers.
- `models/other_feature_scaler.save`: The standardizer of other features.

Two csv files are generated after running `test_model.py`:
- `train_pred.csv`: Calculated and predicted target values of the training set.
- `test_pred.csv`: Calculated and predicted target values of the testing set.

With two `*_pred.csv` files generated, running the `MAE_r.py` prints the performance of the model on the training sets and testing sets, respectively:
```
MAE = 34.7
Pearson r: 0.9988
MAE = 41.3
Pearson r: 0.9987
```
MAEs are in the unit of meV/atom.

## Instruction of Use
### Dataset Preparation
The datasets should contain at least two columns, clusters and training labels, e.g. "AEPA" (Atomization Energy Per Atom):
- Clusters: A string representing the composition and charge of a cluster. A number following after an element symbol represents the number of this element's atoms in the cluster. No number after an element symbol represents one. The end of the string denotes its charge: `+` for cations, `-` for anions, and neither for neutral species. For example, `OsTaHf2O4-` is an anionic cluster containing one Os, one Ta, two Hf, and four O atoms, and is equivalent to `Os1Ta1Hf2O4-`. Only single-charged cations/anions and neutral species are supported.
- Training labels: A float-point number.
- Columns representing atom numbers or charges are not necessary to be included in the datasets, as the model internally interprets them from the clusters' string.
- Using "global features" are also supported in this model. For example, a new column of SMs can be added, obtaining a model predicting AE using composition, charge, AND SM.

### Adopting Atomic Vectors
Atomic vectors of different fineness, i.e. single atomic number, periodic table positions, shell, subshell and orbital, are given in the folder "atomic_vectors". 
To utilize another vector, modify the name of the atomic vector csv file, e.g.:  
```
mv atomprop_subshell.csv atomprop.csv
```
Or modify the lines in `*_model.py`:
```
atom_table = pd.read_csv("./atomprop.csv")
```
### Baseline Model Comparison
In the folder `baseline_models`, four baseline models are given for comparison with CTEN: Random forest (RF), supporting vector regression (SVR), gradient boosting regression tree (GBRT) and multi-layer perceptron (MLP). 

For these models, two kinds of descriptors are adopted: Composition descriptor and orbital-level atomic vectors. Composition descriptors are the numbers of atoms for each element with charge. Orbital-level atomic vectors, denoted as "align" in the scripts, are the concatenation of orbital-level atomic vectors padded with zeroes to form a unified length corresponding to 8 atoms. Charges are also added.

To validate four baseline models using composition descriptor, enter the folder `baseline_models/comp/`, and execute the following scripts in order:
```
python comp_desc.py         # generating composition descriptors for each species
python standardize.py       # standardize the descriptors
python *_pipeline.py        # hyperparameter optimizations and testing
``` 

For using orbital-level atomic vectors, enter the folder `baseline_models/align/`, and execute the following scripts in order:
```
python align_desc.py        # generating "align" descriptors for each species
python standardize.py       # standardize the descriptors
python *_pipeline.py        # hyperparameter optimizations and testing
``` 
### Useful Scripts
Some useful scripts are provided in the folder `useful_scripts`:
- `generating_cluster_list.py`: Generate and write names of all neutral, cationic and anionic clusters with given $N_\mathrm{M}$, whose $N_\mathrm{L}\le N_\mathrm{M}$, into `list.txt`.
- `calc_ae.py`: Calculate AEs from a given list of clusters with their DFT calculated energies. `EZPE.csv` serves as an example of its input file, and `AEPA.csv` will be its output.
- `spin_mult_rounding.py`: Convert the model's output float numbers into integers as the predicted SM for species. `fnsm.csv` serves as an example of its input file, and `phys_sm.csv` will be its output.  

