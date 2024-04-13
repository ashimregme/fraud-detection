import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import warnings

sns.set(palette="Set2")

warnings.filterwarnings("ignore")

# Loading Dataset
data = pd.read_csv('data/paysim.csv')

# Viewing Dataframe
data.tail()

# Checking count of Fraud and Non-Fraud Transactions
print(data['isFlaggedFraud'].value_counts())
print(data['isFraud'].value_counts())

# Checking for Null values in dataset
print(f'null values in dataset: {data.isnull().sum()}')

# Checking for duplicate values in the data
print(f'duplicate values in dataset: {data.duplicated().sum()}')

print('Columns of data')
print(list(data.columns))

# shape of data
print(f'The dataset has shape {data.shape}')

print('Statistical summary of the data')
print('describe: ')
data.describe()
print('info: ')
data.info()

print('Proportion of type of Transactions')
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
