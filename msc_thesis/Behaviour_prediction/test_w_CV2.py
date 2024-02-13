import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)

X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe', 'Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr'], axis=1)
y = df['alloyType']

# Initialize RandomForestClassifier
rf = RandomForestClassifier(max_depth=15, n_estimators=97, random_state=19)

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, random_state=21, shuffle=True)

# Perform stratified k-fold cross-validation
feature_importances = np.zeros(X.shape[1])
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    rf.fit(X_train, y_train)
    feature_importances += rf.feature_importances_

feature_importances /= stratified_kfold.n_splits

# Get feature names
feature_names = X.columns

# Get indices that would sort feature_importances array in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Sort feature importance and feature names in descending order
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Plot feature importance with feature labels in descending order
plt.figure(figsize=(14, 6))
plt.bar(sorted_feature_names, sorted_feature_importances, color='blue')
plt.title('Feature Importance from RandomForestClassifier')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


