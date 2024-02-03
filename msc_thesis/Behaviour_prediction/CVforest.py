import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# StratifiedKFold: Takes class information into account to avoid building folds with imbalanced class distributions 
# (for binary or multiclass classification tasks).
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df = df.drop(8)
df['1-Ti'] = 1 - df['Ti']

#X = df[['1-Ti','T0','Fe_eqnr','e_ortho','dV_ortho']]
X = df[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr','T0','Fe_eqnr','e_HCP','e_ortho','dV_HCP','dV_ortho']]
y = df['alloyType']

n = 0
while n < 20:
    rf = RandomForestClassifier(max_depth = 15, n_estimators = 97) #random_state = 19)
    stratified_kfold = StratifiedKFold(n_splits=5, random_state=21, shuffle=True)

    y_pred = cross_val_predict(rf, X, y, cv=stratified_kfold)
    accuracy = accuracy_score(y, y_pred)
    #print('Accuracy: ', accuracy)

    results = cross_val_score(rf, X, y, cv=stratified_kfold)
    # Output the accuracy and std on the accuracy.
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

    cnf_matrix = confusion_matrix(y, y_pred)
    #cnf_matrix
    non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))
    print(f"Incorrect predictions: {non_diagonal_sum}\n")
    n += 1

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