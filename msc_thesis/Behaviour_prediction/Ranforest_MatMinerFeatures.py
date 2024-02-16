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

m = 0
while m < 10:
    print(f"New epoch!")
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

    max_depth = 15
    n_estimators = 97
    random_state = 19

    n = 0
    while n < 20:
        rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, n_jobs = 1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        cnf_matrix = confusion_matrix(y_test, y_pred)
        #cnf_matrix
        non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))
        print(f"Incorrect predictions: {non_diagonal_sum}")
        # Get the least important feature
        least_important_feature = X_train.columns[np.argmin(rf.feature_importances_)]
        print(f"Least important feature: {least_important_feature}")
        n += 1
    m += 1


class_names = [0,1,2,3]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
dfcnfmatrix = pd.DataFrame(cnf_matrix)

# convert row
row_names = {0:'Martensitic at room T',
             1:'Slip or TWIP',
             2:'Superelastic',
             3:'TRIP'}

dfcnfmatrix = dfcnfmatrix.rename(index = row_names)

# convert column
dfcnfmatrix.rename(columns = {0:'Martensitic at room T', 1:'Slip or TWIP' , 2:'Superelastic' , 3:'TRIP'}, inplace = True) 

sns.heatmap(dfcnfmatrix, annot=True, cmap="YlGnBu" ,fmt='g', xticklabels = True, yticklabels = True)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(14, 6))
feature_importances.plot.bar()
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()