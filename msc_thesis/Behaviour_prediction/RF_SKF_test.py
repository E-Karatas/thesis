import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Read the saved DataFrame
file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti']  # alloying content

# Feature selection, a lot of features still need to be dropped
X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr', 'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho', 'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 'e1_HCP', 'e2_HCP', 'e3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho'], axis=1)
y = df['alloyType']

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

Num_features = X.shape[1]
n_estimators = 97
max_depth = 15
n = 0

while n < Num_features:
    sum_incorrect_predictions = 0
    
    # Fit the scaler to the training data and transform it
    X_scaled = scaler.fit_transform(X)
    
    # Initialize a pandas Series to store feature importances sum for each feature
    feature_importances_sum = pd.Series(dtype=float)

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, n_jobs=1, random_state = 10)

    # Stratified KFold cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 10)

    # Perform cross-validation
    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        rf.fit(X_train, y_train)

        # feature importance
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
        feature_importances_sum = feature_importances_sum.add(feature_importances, fill_value=0)

    # Perform cross-validation
    y_pred = cross_val_predict(rf, X, y, cv = skf)

    cnf_matrix = confusion_matrix(y, y_pred)
    non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))
    print(f"Incorrect predictions: {non_diagonal_sum} with SKF with {X.shape[1]} features")

    # Find the feature with the lowest sum of feature importances
    least_important_feature = feature_importances_sum.idxmin()

    # Drop the least important feature
    X = X.drop(columns=[least_important_feature])

    n += 1

