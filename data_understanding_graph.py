import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

threshold = 4

# Loading Dataset
data = pd.read_csv('data/paysim.csv')
# adding feature type1
data["Type2"] = np.nan  # initializing feature column

# filling feature column
data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'), "Type2"] = "CC"
data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'), "Type2"] = "CM"
data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), "Type2"] = "MC"
data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('M'), "Type2"] = "MM"

data = data[data["Type2"] == "CC"]

print(data[['Type2']].describe())

# Sample a portion of the DataFrame
sampled_df = data.sample(frac=0.1)  # Adjust the fraction as needed

# Calculate the frequency of each value in the 'nameDest' column
value_counts = sampled_df['nameDest'].value_counts()

# Convert value_counts to a NumPy array
value_counts_array = value_counts.values

# Filter the array to remove values below the threshold
filtered_value_counts_array = value_counts_array[value_counts_array >= threshold]

# Create a new pandas Series from the filtered NumPy array
filtered_value_counts = pd.Series(filtered_value_counts_array, index=value_counts.index[value_counts_array >= threshold])

# Plot the filtered frequency distribution
plt.figure(figsize=(10, 6))
filtered_value_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Destination Name')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of nameDest Column (Occurrences >= {})'.format(threshold))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
