import pandas as pd

# Load your big dataset
big_dataset = pd.read_csv('data/paysim.csv')

# Specify the number of rows you want in your random subset
subset_size = 100000

# Take a random subset
random_subset = big_dataset.sample(n=subset_size)

# Save the random subset to a file
random_subset.to_csv('data/paysim_random_subset.csv', index=False)
