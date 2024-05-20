import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# The problematic alloy is already dropped in this dataset
file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti'] #alloying content

X = df.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr', 
             'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho', 'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 
             'e1_HCP', 'e2_HCP', 'e3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho'], axis=1)
y = df['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_norm = scaler_x.fit_transform(X)

max_depth = 7
n_estimators = 201
random_state = 1 

rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, n_jobs = 1, random_state=random_state)
rf.fit(X_norm, y)

# Stratified KFold cross-validator
skf_ran = 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=skf_ran)

y_pred = cross_val_predict(rf, X, y, cv = skf)

cnf_matrix = confusion_matrix(y, y_pred)

non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))

print("incorrect predictions on training set with all features:", non_diagonal_sum)

#class_names = [0,1,2,3]
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)

# create heatmap
dfcnfmatrix = pd.DataFrame(cnf_matrix)

# convert row
row_names = {0:'Martensitic at room T',
             1:'Slip or TWIP',
             2:'Superelastic',
             3:'TRIP'}

dfcnfmatrix = dfcnfmatrix.rename(index = row_names)

# convert column
#dfcnfmatrix.rename(columns = {0:'Martensitic at room T', 1:'Slip or TWIP' , 2:'Superelastic' , 3:'TRIP'}, inplace = True) 

#sns.heatmap(dfcnfmatrix, annot=True, cmap="YlGnBu" ,fmt='g', xticklabels = True, yticklabels = True)
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix, training set, all features', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#plt.show()

feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
#plt.figure(figsize=(10, 14))  # Adjust figure size for horizontal display
#feature_importances.plot.barh()  # Use barh() for horizontal bar plot
#plt.title('Feature Importance')
#plt.xlabel('Importance')
#plt.ylabel('Feature')  # Swap the labels for x and y axes
#plt.tight_layout()
#plt.show()

# %% make predictions on validation set before choosing fewer features to compare performance
file_name = 'validation_set_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_pickle(file_path)

df2['1-Ti'] = 1 - df2['Ti'] #alloying content
# Remove rows with empty values in the 'alloyType' column
df2 = df2.dropna(subset=['alloyType'])

X_bio = df2.drop(['alloyType', 'alloy', 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr', 
             'a_BCC', 'a_HCP', 'b_HCP', 'c_HCP', 'a_ortho', 'b_ortho', 'c_ortho', 'lambda1_HCP', 'lambda2_HCP', 'lambda3_HCP', 
             'e1_HCP', 'e2_HCP', 'e3_HCP', 'lambda1_ortho', 'lambda2_ortho', 'lambda3_ortho', 'e1_ortho', 'e2_ortho', 'e3_ortho'], axis=1)
y_bio = df2['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_bio_norm = scaler_x.fit_transform(X_bio)

# Use the trained CV model to make predictions on the new dataset
y_pred_bio = cross_val_predict(rf, X_bio_norm, y_bio, cv=skf)

bio_cnf_matrix = confusion_matrix(y_bio, y_pred_bio)

bio_non_diagonal_sum = np.sum(bio_cnf_matrix) - np.sum(np.diag(bio_cnf_matrix))

print("incorrect predictions on validation set with all features:", bio_non_diagonal_sum)

# %%
# Select the Z most important features
Z = 17
top_Z_features = feature_importances[:Z].index
X = df[top_Z_features]
y = df['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_norm = scaler_x.fit_transform(X)

rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, n_jobs = 1, random_state=random_state)
rf.fit(X_norm, y)

# Stratified KFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=skf_ran)

y_pred = cross_val_predict(rf, X, y, cv = skf)

cnf_matrix = confusion_matrix(y, y_pred)

non_diagonal_sum = np.sum(cnf_matrix) - np.sum(np.diag(cnf_matrix))

print("incorrect predictions on training set with", Z, "features:", non_diagonal_sum)

# Find misclassified samples
#misclassified_indices = y[y != y_pred].index

# Print the alloy compositions of the misclassified samples
#misclassified_alloys = df.loc[misclassified_indices, 'alloy']
#print(misclassified_alloys)

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
plt.title('Confusion matrix, training set, fewer features', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

file_name = 'validation_set_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_pickle(file_path)

df2['1-Ti'] = 1 - df2['Ti'] #alloying content
# Remove rows with empty values in the 'alloyType' column
df2 = df2.dropna(subset=['alloyType'])

X_bio = df2[top_Z_features]
y_bio = df2['alloyType']

# Create an instance of MinMaxScaler
scaler_x = MinMaxScaler()
X_bio_norm = scaler_x.fit_transform(X_bio)

# Use the trained CV model to make predictions on the new dataset
y_pred_bio = cross_val_predict(rf, X_bio_norm, y_bio, cv=skf)

bio_cnf_matrix = confusion_matrix(y_bio, y_pred_bio)

bio_non_diagonal_sum = np.sum(bio_cnf_matrix) - np.sum(np.diag(bio_cnf_matrix))

print("incorrect predictions on validation set with", Z, "features:", bio_non_diagonal_sum)

# Find misclassified samples
#misclassified_indices = y_bio[y_bio != y_pred_bio].index

# Print the alloy compositions of the misclassified samples
#misclassified_alloys = df2.loc[misclassified_indices, 'alloy']
#print(misclassified_alloys)

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
plt.title('Confusion matrix, validation set, fewer features', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()