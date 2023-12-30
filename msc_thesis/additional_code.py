from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = xg_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot feature importance
import matplotlib.pyplot as plt

feature_importance = xg_clf.feature_importances_
features = X_train.columns
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
