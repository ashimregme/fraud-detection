import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Setting the maximum number of columns to be displayed to None
pd.set_option('display.max_columns', None)


def preprocess_data(file_path, sampling_strategy=0.2):
    """
    This function preprocesses the data for machine learning model.

    Parameters:
    file_path (str): The path to the csv file to be preprocessed.
    sampling_strategy (float): The sampling strategy to use for RandomUnderSampler. Default is 0.2.

    Returns:
    x_res (DataFrame): The resampled training data.
    y_res (Series): The resampled training labels.
    x_train (DataFrame): The original training data.
    x_test (DataFrame): The test data.
    y_train (Series): The original training labels.
    y_test (Series): The test labels.
    """

    # Reading the csv file
    data = pd.read_csv(file_path)

    # Adding a new column "Type2" with default value as NaN
    data["Type2"] = np.nan

    # Populating the "Type2" column based on the conditions
    data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'), "Type2"] = "CC"
    data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'), "Type2"] = "CM"
    data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), "Type2"] = "MC"
    data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('M'), "Type2"] = "MM"

    # Adding a new column "HourOfDay" with default value as NaN
    data["HourOfDay"] = np.nan
    # Populating the "HourOfDay" column with the remainder of step divided by 24
    data.HourOfDay = data.step % 24

    # Printing the head of the dataset
    # print("Head of dataset: \n", pd.DataFrame.head(data))

    # Dropping the unnecessary columns
    data = data.drop(["isFlaggedFraud", 'nameOrig', 'nameDest'], axis=1)

    # One-hot encoding the categorical variables
    data = pd.get_dummies(data, prefix=['type', 'Type2'], drop_first=True)

    # Printing the head of the dataset
    # print("Head of dataset: \n", pd.DataFrame.head(data))

    # Splitting the data into features and labels
    x = data.drop("isFraud", axis=1)
    y = data.isFraud

    # Splitting the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Initializing the MinMaxScaler
    scaler = MinMaxScaler()

    # Scaling the training data
    x_train = scaler.fit_transform(x_train)

    # Scaling the test data
    x_test = scaler.transform(x_test)

    # Initializing the RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)

    # Printing the unique values in y_train
    print(y_train.unique())

    # Resampling the training data
    x_res, y_res = rus.fit_resample(x_train, y_train)

    return x_res, y_res, x_train, x_test, y_train, y_test
