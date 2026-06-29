import pandas as pd
from CTEN import (
    CTEN,
)

atom_table = pd.read_csv("./atomprop.csv")
atom_column_name = "atom"


data_train = pd.read_csv("./train_set.csv")
data_test = pd.read_csv("./test_set.csv")

cluster_column_name = "cluster"

target_column_name = "AEPA"

# create models
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


# load model
model.load(
    "./models/cten.pth",
    "./models/num_atoms_scaler.save",
    "./models/other_feature_scaler.save",
)

# predict the trainning set
train_X = data_train.drop(columns=[target_column_name])
train_pred = model.predict(train_X)
train_pred_df = pd.DataFrame(train_pred)
train_pred_df = pd.concat([data_train, train_pred_df], axis=1)
train_pred_df.columns = ['cluster', 'Calculated AEPA', 'Predicted AEPA']
train_pred_df.to_csv('train_pred.csv', index=False)

# predict the testing set
test_X = data_test.drop(columns=[target_column_name])
test_pred = model.predict(test_X)
test_pred_df = pd.DataFrame(test_pred)
test_pred_df = pd.concat([data_test, test_pred_df], axis=1)
test_pred_df.columns = ['cluster', 'Calculated AEPA', 'Predicted AEPA']
test_pred_df.to_csv('test_pred.csv', index=False)

