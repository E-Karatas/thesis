import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# The problematic alloy is already dropped in this dataset
file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
#print(print(df.columns))

df['1-Ti'] = 1 - df['Ti'] #alloying content
X = df[['1-Ti','T0','Fe_eqnr','e_HCP','e_ortho','dV_HCP','dV_ortho']]
y = df['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_norm = scaler_x.fit_transform(X)

logreg = LogisticRegression(multi_class='multinomial', max_iter=10000)
# fit the model with data
logreg.fit(X_norm, y)

# Stratified KFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

y_pred = cross_val_predict(logreg, X, y, cv = skf)

cnf_matrix = confusion_matrix(y, y_pred)

non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))

print("incorrect predictions: ", non_diagonal_sum)

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

# Get the absolute values of the coefficients and feature names
coef_abs = np.abs(logreg.coef_[0])
feature_names = X.columns

# Get the indices that would sort the coefficient absolute values in ascending order
sorted_indices = np.argsort(coef_abs)

# Rearrange the coefficient absolute values and feature names based on the sorted indices
sorted_coef_abs = coef_abs[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_coef_abs)), sorted_coef_abs, tick_label=sorted_feature_names)
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.title('Feature Importances for Logistic Regression (Sorted)')
plt.show()

# %%

file_name = 'validation_set_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_pickle(file_path)

df2['1-Ti'] = 1 - df2['Ti'] #alloying content
# Remove rows with empty values in the 'alloyType' column
df2 = df2.dropna(subset=['alloyType'])
X_bio = df2[['1-Ti','T0','Fe_eqnr','e_HCP','e_ortho','dV_HCP','dV_ortho']]
y_bio = df2['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_bio_norm = scaler_x.fit_transform(X_bio)

# Use the trained CV model to make predictions on the new dataset
y_pred_bio = cross_val_predict(logreg, X_bio_norm, y_bio, cv=skf)

bio_cnf_matrix = confusion_matrix(y_bio, y_pred_bio)

bio_non_diagonal_sum = np.sum(bio_cnf_matrix) - np.sum(np.diag(bio_cnf_matrix))

print("incorrect predictions on validation set: ", bio_non_diagonal_sum)

class_names = [0,1,2,3]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
bio_dfcnfmatrix = pd.DataFrame(bio_cnf_matrix)

# convert row
row_names = {0:'Martensitic at room T',
             1:'Slip or TWIP',
             2:'Superelastic',
             3:'TRIP'}

dfcnfmatrix = bio_dfcnfmatrix.rename(index = row_names)

# convert column
dfcnfmatrix.rename(columns = {0:'Martensitic at room T', 1:'Slip or TWIP' , 2:'Superelastic' , 3:'TRIP'}, inplace = True) 

sns.heatmap(dfcnfmatrix, annot=True, cmap="YlGnBu" ,fmt='g', xticklabels = True, yticklabels = True)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()