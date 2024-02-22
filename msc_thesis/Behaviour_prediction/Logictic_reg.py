import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df = df.drop(8)

X = df[['Ti','T0','e_ortho','dV_ortho', 'Fe_eqnr']]
y = df['alloyType']

logreg = LogisticRegression(multi_class='multinomial', max_iter=10000)
# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Calculate class distribution
class_distribution = y.value_counts()

# Determine the number of samples for each class in the training and testing sets
train_size = 0.75
test_size = 0.25
train_class_counts = (class_distribution * train_size).round().astype(int)
test_class_counts = (class_distribution * test_size).round().astype(int)

# Initialize empty lists to store indices of samples for training and testing sets
train_indices = []

# Iterate over each class
for label in class_distribution.index:
    # Get indices of samples for the current class
    class_indices = df.index[df['alloyType'] == label].tolist()
    
    # Randomly select indices for training set while maintaining the class distribution
    train_indices.extend(np.random.choice(class_indices, train_class_counts[label], replace=False))

# Remaining indices are for the test set
remaining_indices = df.index.difference(train_indices)

# Randomly select indices for the test set from the remaining indices
test_indices = []
for label in class_distribution.index:
    # Get remaining indices of samples for the current class
    class_remaining_indices = remaining_indices[df.loc[remaining_indices, 'alloyType'] == label]
    
    # Randomly select indices for test set while maintaining the class distribution
    test_indices.extend(np.random.choice(class_remaining_indices, test_class_counts[label], replace=False))

# Create training and testing sets
X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)

non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))

print(non_diagonal_sum)