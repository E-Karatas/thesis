import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

print(file_path)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
#df = df.drop(8), PROBLEMATIC ALLOY ALREADY DROPPED!
print(df.head())

df['1-Ti'] = 1 - df['Ti']

X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr'], axis=1)
print(X.head())
y = df['alloyType']

max_depth = 15
n_estimators = 97
random_state = 19

n = 0
while n < 20:
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 21)

    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, n_jobs = 1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    #cnf_matrix
    non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))
    print(f"Incorrect predictions: {non_diagonal_sum}")
    n += 1
    #random_state += 1

print(rf.classes_)
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

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

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