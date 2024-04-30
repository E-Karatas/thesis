# %%
import numpy as np
import pandas as pd
import os

# %%
file_name = 'df_all.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df = pd.read_parquet(file_path)
df.head()
X = df[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]

file_name = 'bio_df.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df2 = pd.read_parquet(file_path)
#df2.head()
X_bio = df2[['Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']]

# Define the columns of interest
element_columns = ['Nb', 'Mo', 'Zr', 'Fe', 'Ta', 'O']

# Find the index corresponding to the highest value in each column
highest_index = {}
for column in element_columns:
    highest_index[column] = df[df[column] == df[df[column] > 0][column].max()].index[0]

# Find the index corresponding to the lowest non-zero value in each column
lowest_index = {}
for column in element_columns:
    lowest_index[column] = df[df[column] == df[df[column] > 0][column].min()].index[0]

# Print the alloys corresponding to the highest and lowest non-zero values in each column
for column, index in highest_index.items():
    print(f"Alloy with highest {column}: content {df.loc[index, 'alloy']}")

for column, index in lowest_index.items():
    print(f"Alloy with lowest non-zero {column}: content {df.loc[index, 'alloy']}")

