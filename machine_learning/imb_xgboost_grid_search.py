import sys
import warnings

import numpy as np
# %matplotlib inline
import seaborn as sns
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report,
                             roc_auc_score, make_scorer)
from sklearn.model_selection import GridSearchCV

from ml_common import preprocess_data

import time

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv')

start_time = time.time()

# Define the parameter grid
param_grid = {
    'booster': ['dart'],
    'eta': [0.3],
    'imbalance_alpha': [6],
    'special_objective': ['weighted'],
    'max_depth': [5],
    'eval_metric': ['auc'],
    'num_round': [40]
}

# CV_booster = GridSearchCV(
#     estimator=imb_xgb(),
#     cv=5,
#     param_grid=param_grid,
#     scoring=make_scorer(recall_score, average="binary"),
# scoring='average_precision',
# n_jobs=-1
# )

# CV_booster.fit(x_train, y_train.to_numpy())

# Get the best parameters
# best_params = CV_booster.best_params_

# print(f"Best parameters: {best_params}")

# opt_booster = CV_booster.best_estimator_
# 'booster': ['dart'],
# 'eta': [0.3],
# 'imbalance_alpha': [6],
# 'special_objective': ['weighted'],
# 'max_depth': [5],
# 'eval_metric': ['auc'],
# 'num_round': [40]
xg_booster = imb_xgb(
    booster='gbtree',
    eta=0.35,
    imbalance_alpha=10,
    special_objective='weighted',
    # focal_gamma=0.01,
    # special_objective='focal',
    max_depth=8,
    eval_metric=['auc', 'error', 'logloss'],
    num_round=80,
)
xg_booster.fit(x_train, y_train.to_numpy())
y_pred = xg_booster.predict_determine(x_test)

end_time = time.time()

# Evaluating model
print(y_pred[:5])
y_pred = np.where(np.isnan(y_pred), 1, y_pred)
print(y_pred[:5])
CM_RF = confusion_matrix(y_test, y_pred)
CR_RF = classification_report(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_RF)
print("Classification Report:\n", CR_RF)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Average Precision (AP):", average_precision_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

print("Time taken: ", end_time - start_time, "seconds")

sys.exit(0)
