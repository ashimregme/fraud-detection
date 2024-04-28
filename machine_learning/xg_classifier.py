import sys
import warnings

# %matplotlib inline
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report,
                             roc_auc_score)
from xgboost import XGBClassifier

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.01)

# Define the parameter grid
# param_grid = {
#     'max_depth': [2, 3, 4, 5],
#     'learning_rate': [0.2, 0.15, 0.1, 0.05, 0.02, 0.01],
#     'n_estimators': [50, 100, 150, 200, 300, 350],
#     'min_child_weight': [0.1, 0.5, 0.9, 1, 1.5, 2, 3]
# }
param_grid = {
    'learning_rate': [0.2, 0.1, 0.05],
    'n_estimators': [350, 400, 450, 900],
    'min_child_weight': [0.01, 0.1, 0.5]
}

# Initialize the XGBoost classifier
# model = XGBClassifier(use_label_encoder=False, eval_metric=['auc', 'logloss'])

# Initialize the Grid Search model
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc')

# Fit the grid search model
# grid_search.fit(x_res, y_res)

# Get the best parameters
# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")

xgb_model = XGBClassifier(max_depth=5, learning_rate=0.05,
                          n_estimators=700,
                          min_child_weight=1.5,
                          use_label_encoder=False,
                          eval_metric=['auc', 'error', 'logloss'])
# xgb_model = XGBClassifier(max_depth=5, learning_rate=0.15,
#                           n_estimators=350, min_child_weight=0.1,
#                           use_label_encoder=False)

xgb_model.fit(x_res, y_res)

y_pred = xgb_model.predict(x_test)
y_predtrain = xgb_model.predict(x_train)

# Evaluating model
CM_RF_train = confusion_matrix(y_train, y_predtrain)
CR_RF_train = classification_report(y_train, y_predtrain)
CM_RF = confusion_matrix(y_test, y_pred)
CR_RF = classification_report(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_RF)
print("Classification Report:\n", CR_RF)
print("Confusion Matrix Train Data:\n", CM_RF_train)
print("Classification Report Train Data:\n", CR_RF_train)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Average Precision (AP):", average_precision_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

sys.exit(0)
