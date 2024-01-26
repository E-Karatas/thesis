import numpy as np
import pandas as pd
import os
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df = df.drop(8)

X = df[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr','T0','Fe_eqnr','e_HCP','e_ortho','dV_HCP','dV_ortho']]
y = df['alloyType']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

rf = RandomForestClassifier(max_depth = 5, n_estimators = 69)
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

class_names=[0,1,2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# Text(0.5,257.44,'Predicted label');
plt.show()

# cnf_matrix = cnf_matrix(y_test, y_predict)
# sns.heatmap(cnf_matrix, annot=True, fmt="d")