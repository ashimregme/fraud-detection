import sys
import warnings

# %matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report)

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.05)

clf = LogisticRegression()
clf.fit(x_res, y_res)
y_pred = clf.predict(x_test)
y_predtrain = clf.predict(x_train)
CM_LR = confusion_matrix(y_test, y_pred)
CR_LR = classification_report(y_test, y_pred)
CM_LRtrain = confusion_matrix(y_train, y_predtrain)
CR_LRtrain = classification_report(y_train, y_predtrain)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_LR)
print("Classification Report:\n", CR_LR)
print("Confusion Matrix Train:\n", CM_LRtrain)
print("Classification Report Train:\n", CR_LRtrain)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Average Precision (AP):", average_precision_score(y_test, y_pred))

sys.exit(0)
