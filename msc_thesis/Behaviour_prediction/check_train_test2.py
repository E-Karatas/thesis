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

from sklearn.model_selection import StratifiedShuffleSplit

m = 0
while m < 20:
    # Create an instance of StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)

    # Get the training and testing indices
    train_index, test_index = next(sss.split(X, y))

    # Create training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #print(y_train.value_counts())
    #print(y_test.value_counts())

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
    
    print(f"New epoch!")
    m += 1