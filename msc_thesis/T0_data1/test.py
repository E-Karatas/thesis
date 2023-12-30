import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split

df = pd.read_parquet('Labeled_data.parquet')

# Extract feature and target arrays
X, y = df.drop('Deformation_mechanism_simple', axis=1), df[['Deformation_mechanism_simple']]

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')

# Extract text features
cats = y.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   y[col] = y[col].astype('category')

print(X.dtypes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

