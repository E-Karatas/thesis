# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os

# %%
file_name = 'df.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)

# %%
df = df[df['Ms'] != 0]
#print(df)

X = df.drop(['alloy', 'Ms', 'Deformation_mechanism','Deformation_mechanism_simple', 'composition', 'Microstructure'], axis=1)
y = df[['Ms']]

# %%
lr = LinearRegression()
lr.fit(X, y)

# %%
# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))

# %%
from sklearn.model_selection import KFold, cross_val_score

# Use 20-fold cross validation (80% training, 20% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

# %%
from sklearn.model_selection import cross_val_predict

y_pred_cv = cross_val_predict(lr, X, y, cv=crossvalidation)
y_array = y.values

# %%
import matplotlib.pyplot as plt

# Assuming y_array and y_pred_cv are NumPy arrays
y_array = y.values.flatten()  # Flatten the 2D array to 1D
y_pred_cv = y_pred_cv.flatten()
print()

# Plot the true labels (y_array) against the predicted labels (y_pred_cv)
plt.scatter(y_array, y_pred_cv, label='True vs Predicted', color='blue')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True Ms [K]')
plt.ylabel('Predicted Ms [K]')
plt.title('Linear Regression - True vs Predicted Ms')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %%
