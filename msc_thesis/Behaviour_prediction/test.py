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

#print(df.iloc[0])

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

# Create a dataframe with your alloy compositions
alloys = [
    {'Al': 0.0, 'Cr': 0.0, 'Fe': 0.009037, 'Mo': 0.052604, 'Nb': 0.0, 'O': 0.0, 'Ta': 0.0, 'Sn': 0.0, 'Ti': 0.938359, 'W': 0.0, 'V': 0.0, 'Zr': 0.0},
    # Add more alloys as needed
]

df_compositions = pd.DataFrame(alloys)

# Convert wt% to at%
# Add your conversion logic here if needed

# Query the materials database using matminer
mpr = MPDataRetrieval(api_key="YOUR_MP_API_KEY")  # Get your API key from materialsproject.org

# Specify the properties you are interested in
properties = ['material_id', 'K_VRH', 'G_VRH']

# Query the materials project database for bulk modulus (K) and shear modulus (G)
df_materials = mpr.get_all_data(df_compositions, properties)

# Merge the two dataframes based on the material_id
df_result = pd.merge(df_compositions, df_materials, on='material_id')

# Print the resulting dataframe
print(df_result)
