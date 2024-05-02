import sys
import time
import warnings

# %matplotlib inline
import seaborn as sns
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report,
                             roc_auc_score)

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv')

start_time = time.time()

xg_booster = imb_xgb()

xg_booster.fit(x_res, y_res)

y_pred = xg_booster.predict_determine(x_test)

end_time = time.time()

# Evaluating model
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
