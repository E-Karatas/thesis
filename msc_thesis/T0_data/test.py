import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Assuming df is your DataFrame with 'alloy', 'T0', and 'Deformation_mechanism_simple' columns
df = pd.read_parquet('/home/erkan/Desktop/msc_thesis/T0_data/Labeled_data.parquet')

#print(df.head())

print(df.dtypes)