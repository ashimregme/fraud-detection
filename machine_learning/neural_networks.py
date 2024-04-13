import sys
import warnings

import numpy as np
# %matplotlib inline
import seaborn as sns
from sklearn.metrics import (f1_score, confusion_matrix,
                             precision_score, recall_score, classification_report, roc_auc_score)
from tensorflow import keras

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.2)

model = keras.Sequential([
    keras.layers.Dense(15, input_shape=(12,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

Y_pred1 = model.predict(x_test)

Y_pred = np.where(Y_pred1 < 0.44, 0, 1)

y_predtrain1 = model.predict(x_train)

y_predtrain = np.where(y_predtrain1 < 0.38, 0, 1)

CM_NN = confusion_matrix(y_test, Y_pred)
CR_NN = classification_report(y_test, Y_pred)
CM_NNtrain = confusion_matrix(y_train, y_predtrain)
CR_NNtrain = classification_report(y_train, y_predtrain)
ROC_AUC_SCORENN = roc_auc_score(y_test, Y_pred)
print("Confusion Matrix:\n", CM_NN)
print("Classification Report:\n", CR_NN)
print("Confusion Matrix Train:\n", CM_NNtrain)
print("Classification Report Train:\n", CR_NNtrain)
print("Area Under Curve:", ROC_AUC_SCORENN)
print("Precision:", precision_score(y_test, Y_pred))
print("Recall:", recall_score(y_test, Y_pred))
print("F1:", f1_score(y_test, Y_pred))

sys.exit(0)
