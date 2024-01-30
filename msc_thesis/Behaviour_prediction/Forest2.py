import numpy as np
import pandas as pd
import os
# import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df = df.drop(8)

X = df[['T0','e_ortho','dV_ortho']]
y = df['alloyType']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

rf = RandomForestClassifier(max_depth = 15, n_estimators = 97)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

y_predict = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: ', accuracy)

print(rf.classes_)

import matplotlib.pyplot as plt
import seaborn as sns

cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix

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
