import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame with 'alloy', 'T0', and 'Deformation_mechanism_simple' columns
df = pd.read_parquet('/home/erkan/Desktop/msc_thesis/Labeled_data.parquet')

print(df.head())

alloy = df[['alloy']]
X = df.drop(['alloy', 'Deformation_mechanism_simple', 'composition'], axis=1)
y = df[['Deformation_mechanism_simple']]

# Label encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.values.ravel())

# Create XGBClassifier instance
xg_clf = xgb.XGBClassifier(objective='multi:softmax', subsample=0.70, 
                           colsample_bytree=0.60, eta=0.5, max_depth=4, 
                           n_estimators=1000, num_class=len(label_encoder.classes_))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
xg_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Make predictions on the test set
y_pred = xg_clf.predict(X_test)

# Evaluate test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

