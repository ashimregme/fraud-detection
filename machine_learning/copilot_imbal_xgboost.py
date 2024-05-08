# Import necessary libraries
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report,
                             roc_auc_score)
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.model_selection import train_test_split

# Create a binary classification dataset with imbalance
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_classes=2, weights=[0.99, 0.01])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifier
clf = imb_xgb(special_objective='focal', focal_gamma=2.0)

print(X_train.shape)
print(y_train.shape)

array = y_train
# Print the number of dimensions
print("Number of dimensions: ", array.ndim)

# Print the shape of the array
print("Shape of array: ", array.shape)

# Print the size of the array
print("Size of array: ", array.size)

# Print the data type of the array
print("Data type of array: ", array.dtype)

# Print the item size of the array
print("Item size of array: ", array.itemsize)

# Print the data of the array
print("Data of array: ", array.data)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict_determine(X_test)

CM_RF = confusion_matrix(y_test, y_pred)
CR_RF = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", CM_RF)
print("Classification Report:\n", CR_RF)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Average Precision (AP):", average_precision_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))