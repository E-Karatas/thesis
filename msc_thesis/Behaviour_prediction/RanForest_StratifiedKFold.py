import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# The problematic alloy is already dropped in this dataset
file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti'] # alloying content

# Feature selection, a lot of features still need to be dropped 
X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr', 'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho', 'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 'e1_HCP', 'e2_HCP', 'e3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho'], axis=1)
y = df['alloyType']

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler to the training data and transform it
X = scaler.fit_transform(X)

Num_features = X.shape[1]

n_estimators = 97
max_depth = 15
n = 0

rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, n_jobs = 1, random_state = 75)

# Stratified KFold cross-validator 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Perform cross-validation
y_pred = cross_val_predict(rf, X, y, cv = skf)

cnf_matrix = confusion_matrix(y, y_pred)
non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))
print(f"Incorrect predictions: {non_diagonal_sum}")

