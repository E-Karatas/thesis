# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

# %%
file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)

# %%
X = df[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]
y = df[['T0']]#.values.ravel()

scaler_x = MinMaxScaler()
X_norm = scaler_x.fit_transform(X)

scaler_y = MinMaxScaler()
# Reshape y to a 2D array if it's a 1D array
if len(y.shape) == 1:
    y = y.reshape(-1, 1)
y_norm = scaler_y.fit_transform(y)

rf = RandomForestRegressor()
rf.fit(X_norm, y_norm)

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=2)
y_pred_cv = cross_val_predict(rf, X_norm, y_norm, cv=crossvalidation)

# Inverse transform the predicted target variable to return to original scaling
y_pred_cv = scaler_y.inverse_transform(y_pred_cv.reshape(-1, 1))

# Calculate R-squared (R2) score
r2 = r2_score(y, y_pred_cv)
print('R-squared (R2) score:', r2)
# Calculate root mean squared error (RMSE)
mse = mean_squared_error(y, y_pred_cv)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

import matplotlib.pyplot as plt

# Plot the true labels (y_array) against the predicted labels (y_pred_cv)
plt.scatter(y, y_pred_cv, label='True vs Predicted', color='blue')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True T0 [K]')
plt.ylabel('Predicted T0 [K]')
plt.title('Random Forest CV Model - True vs Predicted T0')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %%
file_name = 'validation_set_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_pickle(file_path)

X_bio = df2[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]
y_bio = df2[['T0']]

scaler_x = MinMaxScaler()
X_bio_norm = scaler_x.fit_transform(X_bio)

scaler_y = MinMaxScaler()
# Reshape y to a 2D array if it's a 1D array
if len(y_bio.shape) == 1:
    y_bio = y_bio.reshape(-1, 1)
y_bio_norm = scaler_y.fit_transform(y_bio)

# Use the cross-validated model to make predictions on the new dataset
y_pred_bio = cross_val_predict(rf, X_bio_norm, y_bio_norm, cv=crossvalidation)

# Inverse transform the predicted target variable to return to original scaling
y_pred_bio = scaler_y.inverse_transform(y_pred_bio.reshape(-1, 1))

# Calculate R-squared (R2) score
r2_bio = r2_score(y_bio, y_pred_bio)
print('R-squared (R2) score:', r2_bio)
# Calculate root mean squared error (RMSE)
mse_bio = mean_squared_error(y_bio, y_pred_bio)
rmse_bio = np.sqrt(mse_bio)
print('Root Mean Squared Error (RMSE):', rmse_bio)

# Plot the true labels against the predicted labels for the new dataset
plt.scatter(y_bio, y_pred_bio, label='True vs Predicted (Bio Dataset)', color='red')
plt.plot([0, 1500], [0, 1500], linestyle='--', color='black', label='Ideal line')

# Customize the plot
plt.xlabel('True T0 [K]')
plt.ylabel('Predicted T0 [K]')
plt.title('Random Forest CV Model - True vs Predicted T0 (Bio Dataset)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
