import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

train_df = pd.read_csv('train_aligned.csv')
val_df = pd.read_csv('val_aligned.csv')
test_df = pd.read_csv('test_aligned.csv')

ss = StandardScaler()
col_names= train_df.columns.to_list()
heads = col_names[0:1]
tails = col_names[-1:]

tags = train_df[heads]
labels = train_df[tails]
#print(labels)
feats = train_df.drop(heads + tails, axis=1)
#print(feats)
ss.fit(feats)
joblib.dump(ss, "standard_scaler.save")
X_scaler = ss.transform(feats)
feat_output=pd.DataFrame(X_scaler)
new_train = pd.concat([tags, feat_output, labels], axis=1)
new_train.to_excel("Train_data_for_model_validation.xlsx", index=False)

tags = test_df[heads]
labels = test_df[tails]
feats = test_df.drop(heads + tails, axis=1)
X_scaler = ss.transform(feats)
feat_output=pd.DataFrame(X_scaler)
new_test = pd.concat([tags, feat_output, labels], axis=1)
new_test.to_excel("Test_data_for_model_validation.xlsx", index=False)


tags = val_df[heads]
labels = val_df[tails]
feats = val_df.drop(heads + tails, axis=1)
X_scaler = ss.transform(feats)
feat_output=pd.DataFrame(X_scaler)
new_val = pd.concat([tags, feat_output, labels], axis=1)
new_val.to_excel("Val_data_for_model_validation.xlsx", index=False)
