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

print(df.head())

# %%
