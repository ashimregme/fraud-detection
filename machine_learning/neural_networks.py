import sys
import warnings

import numpy as np
# %matplotlib inline
import seaborn as sns
from sklearn.metrics import (f1_score, confusion_matrix,
                             precision_score, recall_score, classification_report, roc_auc_score)
from tensorflow import keras
from keras.callbacks import EarlyStopping

from ml_common import preprocess_data

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

x_res, y_res, x_train, x_test, y_train, y_test = preprocess_data('../data/paysim.csv', 0.2)

print('x_res.shape: ', x_res.shape)
print('x_train.shape: ', x_train.shape)
print('x_test.shape: ', x_test.shape)

model = keras.Sequential([
    keras.layers.Dense(12, input_shape=(12,), activation='relu'),
    # keras.layers.Dense(75, activation='relu'),
    # keras.layers.Dense(50, activation='relu'),
    # keras.layers.Dense(25, activation='relu'),
    # keras.layers.Dense(12, activation='relu'),
    # keras.layers.Dense(6, activation='relu'),
    # keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
# Define your early stopping criteria
early_stopping = EarlyStopping(monitor='val_Recall', patience=1, min_delta=0.005, mode='auto', verbose=1, restore_best_weights=True)
batch_size = 32
model.fit(x_res, y_res, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test), callbacks=[])

# y_train_pred = model.predict(x_train, batch_size=batch_size)
# y_train_pred_binary = np.argmax(y_train_pred, axis=-1)
# print(np.unique(y_train_pred_binary))
y_test_pred = model.predict(x_test, batch_size=batch_size)
y_test_pred_binary = np.argmax(y_test_pred, axis=-1)

CM_NN = confusion_matrix(y_test, y_test_pred_binary)
CR_NN = classification_report(y_test, y_test_pred_binary)
# CM_NN_train = confusion_matrix(y_train, y_train_pred_binary)
# CR_NN_train = classification_report(y_train, y_train_pred_binary)
ROC_AUC_SCORE_NN = roc_auc_score(y_test, y_test_pred_binary)
print("Confusion Matrix:\n", CM_NN)
print("Classification Report:\n", CR_NN)
# print("Confusion Matrix Train:\n", CM_NN_train)
# print("Classification Report Train:\n", CR_NN_train)
print("ROC AUC score:", ROC_AUC_SCORE_NN)
print("Precision:", precision_score(y_test, y_test_pred_binary))
print("Recall:", recall_score(y_test, y_test_pred_binary))
print("F1:", f1_score(y_test, y_test_pred_binary))

sys.exit(0)
