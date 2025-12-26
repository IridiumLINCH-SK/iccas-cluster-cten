The original code was written by Zi-Yue Wang, whose modification was made by Ning-Zheng Li (Iridium LINCH-SK). 

The CTEN model has following requirements: 
- Python (>= 3.9)
- NumPy
- pandas
- scikit-learn
- SciPy
- PyTorch
- matplotlib

The label used in the training is the opposite number of average atomization energy, which is forming energy per atom (FEPA).

# CTEN
The model. CTEN is the abbreviation of Cluster Transformer Encoder Network.

# cten_examples
An example of utilizing CTEN.
Train and validate the model by running `train_val_model.py`, and test its performance by running `test_model.py`.

# baseline_models
"align" means using orbital-level atomic vectors for baseline models.
Run `align_desc.py` or `comp_desc.py` for the descriptors.
Run `Standardize.py` for standardization.
Run `*_pipeline.py` for hyperparameter optimizations and testing.

# atomic_vectors
Atomic vectors of different fineness, i.e. single atomic number, periodic table positions, shell, subshell and orbital.
