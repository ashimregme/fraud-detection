import sys
import warnings

# %matplotlib inline
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report)

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.2)

RF = RandomForestClassifier(n_estimators=15, oob_score=True, n_jobs=-1)

RF.fit(x_res, y_res)

y_pred = RF.predict(x_test)
y_predtrain = RF.predict(x_train)

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
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

sys.exit(0)
