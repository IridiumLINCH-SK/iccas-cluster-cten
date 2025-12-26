from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import torch
import torch.nn as nn
import pandas as pd
import os
import random
import numpy as np
from scipy.stats import uniform, randint, loguniform
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df1 = pd.read_excel("Train_data_for_model_validation.xlsx")
df2 = pd.read_excel("Val_data_for_model_validation.xlsx")
df3 = pd.read_excel("Test_data_for_model_validation.xlsx")

y_train = df1['FEPA']
X_train = df1.drop(columns=['cluster', 'FEPA'])
y_val = df2['FEPA']
X_val = df2.drop(columns=['cluster', 'FEPA'])
y_test = df3['FEPA']
X_test = df3.drop(columns=['cluster', 'FEPA'])
input_dim = len(X_train.columns)


X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
test_fold = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    def __len__(self):
        return len(self.X)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    
    return g

g = setup_seed(42)


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.001, epochs=800, input_dim=input_dim, hidden_layers=[512, 256, 128, 64]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None  

    def _build_model(self):
        model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[1], self.hidden_layers[2]),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[2], self.hidden_layers[3]),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[3], 1)
                ).to(self.device)
        return model

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)

        train_dataset = dataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            generator=g,  
        )

        self.model_ = self._build_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            for bat in train_loader:
                X_tensor, y_tensor = bat
                optimizer.zero_grad()
                y_pred = self.model_(X_tensor)
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()
        return self
    
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = self.model_(X_tensor).cpu().numpy()
        return y_pred.flatten()


param_distributions = {
    'lr': uniform(0.0001, 0.1),
    'epochs': randint(0, 1600),
}

random_search = RandomizedSearchCV(
    estimator=TorchRegressor(),
    param_distributions=param_distributions,
    n_iter=50,
    cv=ps,
    scoring='neg_mean_squared_error',
    n_jobs=1,
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

best_model = random_search.best_estimator_

# 在各个数据集上评估
datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
}

print("\n=== Model Performance ===")
for name, (X_data, y_data) in datasets.items():
    y_pred = best_model.predict(X_data)
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
