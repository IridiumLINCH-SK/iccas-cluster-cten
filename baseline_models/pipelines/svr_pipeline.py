import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.svm import SVR
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df1 = pd.read_excel("Train_data_for_model_validation.xlsx")
df2 = pd.read_excel("Val_data_for_model_validation.xlsx")
df3 = pd.read_excel("Test_data_for_model_validation.xlsx")

index_tags = ["cluster"]
label_tags = ["FEPA"]

X_train = df1.drop(columns=index_tags+label_tags)
y_train = np.array(df1["FEPA"])
X_train = X_train.values

X_val = df2.drop(columns=index_tags+label_tags)
y_val = np.array(df2["FEPA"])
X_val = X_val.values

X_test = df3.drop(columns=index_tags+label_tags)
y_test = np.array(df3["FEPA"])
X_test = X_test.values

X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
test_fold = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold)

svr = SVR()
param_distributions = {
    'C': loguniform(1e-1, 1e3),
    'epsilon': uniform(0.01, 0.5),
    'gamma': loguniform(1e-4, 1e0),
    'kernel': ['rbf', 'linear'],
}

random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_distributions,
    n_iter=50,
    cv=ps,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=2,
    return_train_score=True
)

print("Start hyperparameter optimizing...")
random_search.fit(X_train_val, y_train_val)

with open("summary.txt", 'a') as g:
    g.write(f"Best parameters: {random_search.best_params_}\n")
    g.write(f"Best Score: {random_search.best_score_:.4f}\n")
    g.write(f"Best RMSE: {np.sqrt(-random_search.best_score_):.4f}\n")

best_svr = random_search.best_estimator_

# 在各个数据集上评估
datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
}

print("\n=== Model Performance ===")
for name, (X_data, y_data) in datasets.items():
    y_pred = best_svr.predict(X_data)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_excel("y_{}_pred.xlsx".format(name))
    mse = mean_squared_error(y_data, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_data, y_pred)
    r2 = r2_score(y_data, y_pred)
                                
    with open("summary.txt", 'a') as g:
        g.write(f"\n{name}:\n")
        g.write(f"  MSE: {mse:.4f}\n")
        g.write(f"  RMSE: {rmse:.4f}\n")
        g.write(f"  MAE: {mae:.4f}\n")
        g.write(f"  R_square: {r2:.4f}\n")
