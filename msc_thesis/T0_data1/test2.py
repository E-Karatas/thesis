import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assuming df is your DataFrame with 'alloy', 'T0', and 'Deformation_mechanism_simple' columns
df = pd.read_parquet('Labeled_data.parquet')

# Separate features (X) and target variable (y)
X = df[['alloy', 'T0']]
y = df['Deformation_mechanism_simple']

# Label encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode the 'alloy' column
column_transformer = ColumnTransformer(
    transformers=[
        ('alloy_onehot', OneHotEncoder(), ['alloy'])
    ],
    remainder='passthrough'
)

X_transformed = column_transformer.fit_transform(X)

# Create XGBClassifier instance
xg_clf = xgb.XGBClassifier(objective='multi:softmax', subsample=0.7, colsample_bytree=0.65, eta=0.4, max_depth=4, n_estimators=3000, num_class=len(label_encoder.classes_))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Fit the model
xg_clf.fit(X_train, y_train)

# Make predictions on a new dataset
# Assuming df_new is your DataFrame with 'alloy' and 'T0' columns for prediction
df_new = pd.read_parquet('Unlabeled_data.parquet')

# The target variable 'Deformation_mechanism_simple' is empty, so we drop it
df_new = df_new.drop('Deformation_mechanism_simple', axis=1)

X_new = column_transformer.transform(df_new)

#predictions = xg_clf.predict(X_new)
