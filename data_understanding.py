import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score,
                             classification_report, roc_curve, auc)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from xgboost import XGBClassifier, plot_importance

import warnings

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

# Loading Dataset
data = pd.read_csv('data/paysim_random_subset.csv')

# Viewing Dataframe
data.tail()

# Checking count of Fraud and Non-Fraud Transactions
data['isFlaggedFraud'].value_counts()
data['isFraud'].value_counts()

# Checking for Null values in dataset
data.isnull().sum()

# There are no null values in the data
data.duplicated().sum()

# There are no duplicate values in the data
print(list(data.columns))

# shape of data
print(f'The dataset has shape {data.shape}')

# Statistical summary of the data
data.describe()
data.info()

# Proportion of type of Transactions
typo_of_trans = data['type'].value_counts()
transaction = typo_of_trans.index
count = typo_of_trans.values

# drawing pie-chart
plt.figure(figsize=(8, 8))
plt.pie(count, labels=transaction, autopct='%1.0f%%')
plt.legend(loc='upper left')
plt.show()

# Count of Fraud and Non-Fraud Transactions in each type of Transaction
plt.figure(figsize=(12, 8))
ax = sns.countplot(x="type", hue="isFraud", data=data)
plt.title('Types of Transaction nonFraud and Fraud')
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.1, p.get_height() + 50))

# - Transaction types TRANSFER and CASH_OUT only have FRAUD Transactions.
# - DEBIT Transactions are only few and TRANSFER type Transactions are also less compared to other type of transactions.
# - No of Fraud transactions of TRANSFER type are very close to No of CASH_OUT FRAUD transactions. Probably modus
#   operadi of FRAUD transactions is by transferring funds to a specific account and then CASHING OUT from those accounts.

data1 = data[(data['isFraud'] == 1) & (data['type'] == 'TRANSFER')]
data1['nameDest'].value_counts()

data2 = data[(data['isFraud'] == 1) & (data['type'] == 'CASH_OUT')]
data2['nameOrig'].value_counts()

# There are no specific accounts from which fraud transactions are carried out . Hence we disregard our suspicioun. So these account name features are not useful for us in modelling as is.

fraud = data[data["isFraud"] == 1]
valid = data[data["isFraud"] == 0]

fraud_transfer = fraud[fraud["type"] == "TRANSFER"]
fraud_cashout = fraud[fraud["type"] == "CASH_OUT"]

# checking if the recipient account of a fraudulent transfer was used as a sending account for cashing out
fraud_transfer.nameDest.isin(fraud_cashout.nameOrig).any()

#

# For fraudulent transactions, the account that received funds during a transfer was not used at all for cashing out.

# We derive a new feature Transaction Type2 from these features account types "C" (customer) and "M" (merchant), which would be the first character for each value under nameOrig and nameDest.

# We will create a categorical variable with levels "CC" (Customer to Customer), "CM" (Customer to Merchant), "MC" (Merchant to Customer), "MM" (Merchant to Merchant).

# adding feature type1
data_new = data.copy()  # creating copy of dataset in case I need original dataset
data_new["Type2"] = np.nan  # initializing feature column

# filling feature column
data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'), "Type2"] = "CC"
data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'), "Type2"] = "CM"
data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), "Type2"] = "MC"
data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('M'), "Type2"] = "MM"

# Visualizing Numeric variables against Fraud using Boxplots

plt.figure(figsize=(25, 16))
plt.subplot(2, 2, 1)
sns.boxplot(x='isFraud', y='step', data=data_new)
plt.title('step vs Fraud', fontweight="bold", size=20)
plt.subplot(2, 2, 2)
sns.boxplot(x='isFraud', y='amount', data=data_new, palette="husl")
plt.title('amount vs Fraud', fontweight="bold", size=20)
plt.subplot(2, 2, 3)
sns.boxplot(x='isFraud', y='oldbalanceOrg', data=data_new, palette='husl')
plt.title('oldbalanceOrig vs Fraud', fontweight="bold", size=20)
plt.subplot(2, 2, 4)
sns.boxplot(x='isFraud', y='oldbalanceDest', data=data_new, palette="bright")
plt.title('oldbalanceDest vs Fraud', fontweight="bold", size=20)

#

fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]

print("Fraud transactions by type1: \n", fraud.Type2.value_counts())
print("\n Valid transactions by type1: \n", valid.Type2.value_counts())

# Looking balances before and after the transaction
wrong_orig_bal = sum(data["oldbalanceOrg"] - data["amount"] != data["newbalanceOrig"])
wrong_dest_bal = sum(data["newbalanceDest"] + data["amount"] != data["newbalanceDest"])
print("Percentage of observations with balance errors in the account giving money: ",
      100 * round(wrong_orig_bal / len(data), 2))
print("Percentage of observations with balance errors in the account receiving money: ",
      100 * round(wrong_dest_bal / len(data), 2))

# Looking at Time


bins = 50

valid.hist(column="step", color="green", bins=bins)
plt.xlabel("1 hour time step")
plt.ylabel("# of transactions")
plt.title("# of valid transactions over time")

fraud.hist(column="step", color="red", bins=bins)
plt.xlabel("1 hour time step")
plt.ylabel("# of transactions")
plt.title("# of fraud transactions over time")

plt.tight_layout()
plt.show()

#

num_days = 7
num_hours = 24
fraud_days = (fraud.step // num_hours) % num_days
fraud_hours = fraud.step % num_hours
valid_days = (valid.step // num_hours) % num_days
valid_hours = valid.step % num_hours

# plotting scatterplot of the days of the week, identifying the fraudulent transactions (red) from the valid transactions (green)
plt.subplot(1, 2, 1)
fraud_days.hist(bins=num_days, color="red")
plt.title('Fraud transactions by Day')
plt.xlabel('Day of the Week')
plt.ylabel("# of transactions")

plt.subplot(1, 2, 2)
valid_days.hist(bins=num_days, color="green")
plt.title('Valid transactions by Day')
plt.xlabel('Day of the Week')
plt.ylabel("# of transactions")

plt.tight_layout()
plt.show()

#

plt.subplot(1, 2, 1)
fraud_hours.hist(bins=num_hours, color="red")
plt.title('Fraud transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel("# of transactions")

plt.subplot(1, 2, 2)
valid_hours.hist(bins=num_hours, color="green")
plt.title('Valid transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel("# of transactions")

plt.tight_layout()
plt.show()

#


data_new["HourOfDay"] = np.nan  # initializing feature column
data_new.HourOfDay = data_new.step % 24

print("Head of dataset1: \n", pd.DataFrame.head(data_new))

#


data_new = data_new.drop(["isFlaggedFraud", 'nameOrig', 'nameDest'], axis=1)

# Handling Categorical Variables
data_new = pd.get_dummies(data_new, prefix=['type', 'Type2'], drop_first=True)

# Train-Test Split Standardizing Data
X = data_new.drop("isFraud", axis=1)
y = data_new.isFraud
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Normalizing data so that all variables follow the same scale (0 to 1)
scaler = MinMaxScaler()

# Fit only to the training data
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Model Selection
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.2)
print(y_train.unique())
X_res, y_res = rus.fit_resample(X_train, y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)
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
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

# Bernoulli Naive Bayes
clf = BernoulliNB()
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)
CM_NB = confusion_matrix(y_test, y_pred)
CR_NB = classification_report(y_test, y_pred)
CM_NBtrain = confusion_matrix(y_train, y_predtrain)
CR_NBtrain = classification_report(y_train, y_predtrain)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_NB)
print("Classification Report:\n", CR_NB)
print("Confusion Matrix Train:\n", CM_NBtrain)
print("Classification Report Train:\n", CR_NBtrain)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

# Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)
CM_MNB = confusion_matrix(y_test, y_pred)
CR_MNB = classification_report(y_test, y_pred)
CM_MNBtrain = confusion_matrix(y_train, y_predtrain)
CR_MNBtrain = classification_report(y_train, y_predtrain)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", CM_MNB)
print("Classification Report:\n", CR_MNB)
print("Confusion Matrix Train:\n", CM_MNBtrain)
print("Classification Report Train:\n", CR_MNBtrain)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

# Stochastic Gradient Classifier
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)
SGDClassifier(max_iter=5)
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
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

# RandomForestClassifier
RF = RandomForestClassifier(n_estimators=15, oob_score=True, n_jobs=-1)

RF.fit(X_res, y_res)

y_pred = RF.predict(X_test)
y_predtrain = RF.predict(X_train)

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

# Artificial Neural Networks
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(15, input_shape=(12,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

Y_pred1 = model.predict(X_test)

Y_pred = np.where(Y_pred1 < 0.44, 0, 1)

y_predtrain1 = model.predict(X_train)

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
