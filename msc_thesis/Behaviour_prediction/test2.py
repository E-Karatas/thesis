import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df = df.drop(8)
df['1-Ti'] = 1 - df['Ti']

#X = df[['1-Ti','T0','Fe_eqnr','e_ortho','dV_ortho']]
X = df[['1-Ti','T0','Fe_eqnr','e_HCP','e_ortho','dV_HCP','dV_ortho']]
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
    n += 1
    #random_state += 1

#print(rf.classes_)
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