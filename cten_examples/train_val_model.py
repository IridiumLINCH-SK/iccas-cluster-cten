import pandas as pd
from CTEN import CTEN
from CTEN import set_random_seed

# Load atomic vectors
atom_table = pd.read_excel("./atomprop.xlsx")
atom_column_name = "Element"
# Load Data Sets, Training and Validation
data_train = pd.read_csv("./train_set.csv")
data_val = pd.read_csv("./val_set.csv")

cluster_column_name = "cluster"
react_rate_column_name = "FEPA"

# Set the random seed
set_random_seed(0)

model = CTEN(
    atom_table,
    atom_column_name,
    cluster_column_name,
    num_heads=4,
    layer_norm_eps=1e-6,
    num_transformer_layers=2,
    fnn_layers=[512, 256, 128, 64],
    transformer_dropout=0.1,
    fnn_dropout=0.1,
)


# Train and validate the model in each epoch, the parameters are reverted to the epoch when the validation MAE is minimized.
train_X = data_train.drop(columns=[react_rate_column_name])
train_y = data_train[[react_rate_column_name]]
train_y.insert(1, 'Uplimit (F=1orT=0)', 1)

val_X = data_val.drop(columns=[react_rate_column_name])
val_y = data_val[[react_rate_column_name]]
val_y.insert(1, 'Uplimit (F=1orT=0)', 1)


model.fit(
    train_X,
    train_y,
    8,
    learning_rate=0.001,
    X_val=val_X,
    y_val=val_y,
    record_file="loss_change.csv",
)

# Saving the model and standarizer
model.save(
    "./models/",
    "cten",
    "num_atoms_scaler",
    "other_feature_scaler",
)

