# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os

# %%
file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df.head()

# %%
X = df[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]
y = df[['T0']].values.ravel()
print(X.head())

# %%
lr = LinearRegression()
lr.fit(X, y)

# %%
# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))

# %%
from sklearn.model_selection import KFold, cross_val_score

# Use 5-fold cross validation (80% training, 20% test)
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

# %%
from sklearn.model_selection import cross_val_predict

y_pred_cv = cross_val_predict(lr, X, y, cv=crossvalidation)

# %%
import matplotlib.pyplot as plt

# Plot the true labels (y_array) against the predicted labels (y_pred_cv)
plt.scatter(y, y_pred_cv, label='True vs Predicted', color='blue')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True T0 [K]')
plt.ylabel('Predicted T0 [K]')
plt.title('Linear Regression CV Model - True vs Predicted T0')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %%
file_name = 'bio_df.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_parquet(file_path)
df2.head()

X_bio = df2[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]
y_bio = df2[['T0']].values.ravel()

# Use the trained model to make predictions on the new dataset
y_pred_bio = lr.predict(X_bio)

# Calculate R2 and RMSE for the new predictions
r2_bio = lr.score(X_bio, y_bio)
rmse_bio = np.sqrt(mean_squared_error(y_true=y_bio, y_pred=y_pred_bio))

# Print the results
print('R2 for the new dataset: %.3f' % r2_bio)
print('RMSE for the new dataset: %.3f' % rmse_bio)

# Plot the true labels against the predicted labels for the new dataset
plt.scatter(y_bio, y_pred_bio, label='True vs Predicted (Bio Dataset)', color='red')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True T0 [K]')
plt.ylabel('Predicted T0 [K]')
plt.title('Linear Regression - True vs Predicted T0 (Bio Dataset)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Fit the linear regression model using cross-validation
lr_cv = LinearRegression()
lr_cv.fit(X, y)

# Use the cross-validated model to make predictions on the new dataset
y_pred_bio_cv = cross_val_predict(lr_cv, X_bio, y_bio, cv=crossvalidation)

# Calculate R2 and RMSE for the new predictions
r2_bio_cv = lr_cv.score(X_bio, y_bio)
rmse_bio_cv = np.sqrt(mean_squared_error(y_true=y_bio, y_pred=y_pred_bio_cv))

# Print the results
print('R2 for the new dataset using cross-validated model: %.3f' % r2_bio_cv)
print('RMSE for the new dataset using cross-validated model: %.3f' % rmse_bio_cv)

# Plot the true labels against the predicted labels for the new dataset
plt.scatter(y_bio, y_pred_bio_cv, label='True vs Predicted (Bio Dataset - CV Model)', color='green')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True T0 [K]')
plt.ylabel('Predicted T0 [K]')
plt.title('Linear Regression - True vs Predicted T0 (Bio Dataset - CV Model)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
