from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_ordinal(dataframe, *column_names):
    """
This function takes two arguments (dataframe & column_name) and
creates a new column by assigning ordinal number based on
cases in the dataset.
    """
    for column_name in column_names:
        dataframe[column_name +
                  '_ordinal'], _ = dataframe[column_name].factorize()

df = pd.read_csv('data/paysim.csv')
df[''] = np.arange(1, len(df) + 1)

# Inserting a serial number column at the beginning
df.insert(0, 'serial', range(1, len(df) + 1))

print(df)

print(df['amount'].sort_values())

df['amount_category'] = pd.cut(df['amount'],
                               [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
                                1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000,
                                1900000, 2000000, 4000000, float(
                                   "inf")],
                               labels=['0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K',
                                       '500K-600K', '600K-700K', '700K-800K', '800K-900K', '900K-1M', '1M-1.1M',
                                       '1.1M-1.2M', '1.2M-1.3M', '1.3M-1.4M', '1.4M-1.5M', '1.5M-1.6M', '1.6M-1.7M',
                                       '1.7M-1.8M', '1.8M-1.9M', '1.9M-2.0M', '2.0M-4.0M', 'Above 4.0M'])

print(df[['type', 'isFraud']])
fig, ax = plt.subplots(figsize=(15, 25))
bar = df[['serial', 'type', 'isFraud', 'amount_category']][df['isFraud'] == 1].groupby(
    ['amount_category', 'isFraud']).count().plot.bar(
    title='Fraud distributions of transactions',
    legend=True,
    ax=ax,
)
bar.set_xlabel('Is Fraud?')
bar.set_ylabel('Number of transactions')
plt.show()
to_ordinal(df, 'type')

print(df[['type', 'type_ordinal']])

pie = df.groupby(['type'])[''].count().plot(
    kind='pie', title='Fraud by transaction type', autopct='%1.0f%%'
)
plt.show()
#
# line = df[['', 'duration']].plot.scatter(
#     x='duration',
#     y='',
#     s=1
# )
# line.set_xticks(range(0, 5400, 300))
# plt.show()
#
# # Create a sample dataframe
# df = pd.DataFrame({'a': [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6],
#                    'b': [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86],
#                    'c': [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6],
#                    'd': [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]})
#
# # Calculate the correlation between column 'a' and all other columns
# corr = df.corr()['a'].drop('a')
#
# # Plot the correlation as a bar plot
# corr.plot.bar(title='Correlation between column "a" and other columns',
#               xlabel='Columns', ylabel='Correlation coefficient',
#               legend=True)

# Show the plot
