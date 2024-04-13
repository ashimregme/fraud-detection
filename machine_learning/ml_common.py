import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set the option to display all columns
pd.set_option('display.max_columns', None)

def preprocess_data(file_path, sampling_strategy=0.2):
    # Loading Dataset
    data = pd.read_csv(file_path)

    # adding feature type1
    data_new = data.copy()  # creating copy of dataset in case I need original dataset
    data_new["Type2"] = np.nan  # initializing feature column

    # filling feature column
    data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'), "Type2"] = "CC"
    data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'), "Type2"] = "CM"
    data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), "Type2"] = "MC"
    data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('M'), "Type2"] = "MM"

    data_new["HourOfDay"] = np.nan  # initializing feature column
    data_new.HourOfDay = data_new.step % 24

    print("Head of dataset: \n", pd.DataFrame.head(data_new))

    data_new = data_new.drop(["isFlaggedFraud", 'nameOrig', 'nameDest'], axis=1)

    # Handling Categorical Variables
    data_new = pd.get_dummies(data_new, prefix=['type', 'Type2'], drop_first=True)

    print("Head of dataset: \n", pd.DataFrame.head(data_new))

    # Train-Test Split Standardizing Data
    x = data_new.drop("isFraud", axis=1)
    y = data_new.isFraud
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Normalizing data so that all variables follow the same scale (0 to 1)
    scaler = MinMaxScaler()

    # Fit only to the training data
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    # Model Selection
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    print(y_train.unique())
    x_res, y_res = rus.fit_resample(x_train, y_train)

    return x_res, y_res, x_train, x_test, y_train, y_test
