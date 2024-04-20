import sys
import warnings

import numpy as np
# %matplotlib inline
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, classification_report)

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.05)
print("Number of rows in resampled set: ", x_res.shape[0])
print("Max Iterations: ", int(np.ceil(10 ** 6 / x_res.shape[0])))

# Sci-Kit: Empirically, we found that SGD converges after observing approx. 10^6 training samples.
# Thus, a reasonable first guess for the number of iterations is n_iter = np.ceil(10**6 / n),
# where n is the size of the training set.
clf = SGDClassifier(loss="modified_huber", penalty="elasticnet", max_iter=int(np.ceil(10 ** 6 / x_res.shape[0])))
clf.fit(x_res, y_res)
y_pred = clf.predict(x_test)
y_predtrain = clf.predict(x_train)
CM_svc = confusion_matrix(y_test, y_pred)
CR_svc = classification_report(y_test, y_pred)
CM_svctrain = confusion_matrix(y_train, y_predtrain)
CR_svctrain = classification_report(y_train, y_predtrain)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_svc)
print("Classification Report:\n", CR_svc)
print("Confusion Matrix Train:\n", CM_svctrain)
print("Classification Report Train:\n", CR_svctrain)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Average Precision (AP):", average_precision_score(y_test, y_pred))

sys.exit(0)
