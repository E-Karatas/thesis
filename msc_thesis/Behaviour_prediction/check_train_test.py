import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti'] #alloying content

# feature selection, a lot of features still need to be dropped 
X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr', 'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho', 'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 'e1_HCP', 'e2_HCP', 'e3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho'], axis=1)
y = df['alloyType']

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

# Check for duplicates in X_train
train_duplicates = X_train.index.duplicated().any()

# Check for duplicates in X_test
test_duplicates = X_test.index.duplicated().any()

if train_duplicates:
    print("Duplicates found in X_train.")
else:
    print("No duplicates found in X_train.")

if test_duplicates:
    print("Duplicates found in X_test.")
else:
    print("No duplicates found in X_test.")

# Check if any index values from X_test are already represented in X_train
intersection = X_test.index.intersection(X_train.index)

if not intersection.empty:
    print("Some indexes from X_test are already represented in X_train.")
    print("Intersecting indexes:", intersection)
else:
    print("No indexes from X_test are already represented in X_train.")


from sklearn.model_selection import StratifiedShuffleSplit

# Create an instance of StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

# Get the training and testing indices
train_index2, test_index2 = next(sss.split(X, y))

# Create training and testing sets
X_train, X_test = X.iloc[train_index2], X.iloc[test_index2]
y_train2, y_test2 = y.iloc[train_index2], y.iloc[test_index2]

print(y_train2.value_counts())
print(y_test2.value_counts())