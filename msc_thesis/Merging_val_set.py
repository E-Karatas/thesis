import numpy as np
import pandas as pd
import os

file_name = 'bio_df.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the parquet file
df1 = pd.read_parquet(file_path)
df1 = df1.drop(['composition'], axis=1)

print(len(df1))

file_name = 'validation.parquet'
file_path = os.path.join('msc_thesis', 'data', file_name)

df2 = pd.read_parquet(file_path)

print(len(df2))

# Find non-matching indices
non_matching_indices = df1[~df1['alloy'].isin(df2['alloy'])].index

# Add rows from df1 to df2 based on non-matching indices
df2 = pd.concat([df2, df1.loc[non_matching_indices]], ignore_index=True)

print(len(df2))

alloyType_columns_df2 = df2['alloyType'].notnull().sum()

print("Number of columns with alloyType in df2:", alloyType_columns_df2)

# Specify the columns to check for duplicates
columns_to_check = ['Al', 'Cr', 'Fe', 'Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr']

# Check for duplicate rows based on the specified columns
duplicate_rows = df2.duplicated(subset=columns_to_check, keep=False)

# Filter out rows where alloyType is not specified
has_alloyType = df2['alloyType'].notnull()

# Drop duplicate rows where alloyType is not specified
df2 = df2[~(duplicate_rows & ~has_alloyType)]

print(len(df2))

alloyType_columns_df2 = df2['alloyType'].notnull().sum()

print("Number of columns with alloyType in df2:", alloyType_columns_df2)

print(len(df2))

class_distribution = df2['alloyType'].value_counts()
print(class_distribution)

# Save the dataframe to a file
#df2.to_pickle('validation_set.pkl')

