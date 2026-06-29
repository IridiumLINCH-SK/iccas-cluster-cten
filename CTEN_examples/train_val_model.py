import pandas as pd
from CTEN import CTEN, set_random_seed

atom_table = pd.read_csv("./atomprop.csv")
atom_column_name = "atom"

data_train = pd.read_csv("./train_set.csv")
data_val = pd.read_csv("./val_set.csv")

cluster_column_name = "cluster"

target_column_name = "AEPA"

set_random_seed(0)

# Create Model
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
    device="cuda:1"
)


# Training
train_X = data_train.drop(columns=[target_column_name])
train_y = data_train[[target_column_name]]

val_X = data_val.drop(columns=[target_column_name])
val_y = data_val[[target_column_name]]

model.fit(
    train_X,
    train_y,
    800,
    learning_rate=0.001,
    X_val=val_X,
    y_val=val_y,
    record_file="loss_change.csv",
)

# Save model and standardizers
model.save(
    "./models/",
    "cten",
    "num_atoms_scaler",
    "other_feature_scaler",
)

