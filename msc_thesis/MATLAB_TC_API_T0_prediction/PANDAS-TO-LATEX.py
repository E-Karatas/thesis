import pandas as pd
import os

# The problematic alloy is already dropped in this dataset
# file_name = 'validation_set_matminer.pkl'
file_name = 'df_matminer.pkl'
file_path = os.path.join('msc_thesis', 'data', file_name)

# Read the saved DataFrame
df = pd.read_pickle(file_path)
df['1-Ti'] = 1 - df['Ti'] # alloying content

class_distribution = df['alloyType'].value_counts()
print(class_distribution)
