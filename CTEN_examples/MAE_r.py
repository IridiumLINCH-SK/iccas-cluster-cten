from scipy.stats import pearsonr
import pandas as pd
import numpy as np

train_result = pd.read_csv("train_pred.csv")
c1 = train_result["Calculated AEPA"]
c2 = train_result["Predicted AEPA"]

c1 = np.array(c1)
c2 = np.array(c2)

print("MAE = {:.1f}".format(1000 * np.mean(np.abs(c1 - c2))))
print("Pearson r: {:.4f}".format(pearsonr(c1, c2).statistic))

test_result = pd.read_csv("test_pred.csv")
c1 = test_result["Calculated AEPA"]
c2 = test_result["Predicted AEPA"]
c1 = np.array(c1)
c2 = np.array(c2)

print("MAE = {:.1f}".format(1000 * np.mean(np.abs(c1 - c2))))
print("Pearson r: {:.4f}".format(pearsonr(c1, c2).statistic))
