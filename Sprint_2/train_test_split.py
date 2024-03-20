# imports

import os

import pandas as pd

from sklearn.model_selection import train_test_split


def train_test_split_custom(df: pd.DataFrame, test_size: float=0.2, chosed_dataset: str='BPI_Challenge_2017', lags: bool=False):
    """
    Returns Train part and Test part in mentioned order
    """

    # Split the DataFrame into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    # Identify common 'case:concept:name' values between train and test sets
    common_cases = set(train_df['case:concept:name']).intersection(set(test_df['case:concept:name']))

    # Remove rows with common 'case:concept:name' values from both train and test sets
    train_df = train_df[~train_df['case:concept:name'].isin(common_cases)]
    test_df = test_df[~test_df['case:concept:name'].isin(common_cases)]

    # Convert 'time:timestamp' to datetime format
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Filter events in train set based on 'time:timestamp'
    train_df = train_df[train_df['time:timestamp'] <= test_df['time:timestamp'].min()]

    # Filter events in test set based on 'time:timestamp'
    test_df = test_df[test_df['time:timestamp'] >= train_df['time:timestamp'].max()]

    # Print the resulting DataFrames

    print(f"Train set ends with {train_df['time:timestamp'].max()}")

    print(f"Test set starts with: {test_df['time:timestamp'].min()}")

    if lags:
        train_df['concept:name - lag_1'] = train_df.groupby('case:concept:name')['concept:name'].shift(1).fillna('absent')
        train_df['concept:name - lag_2'] = train_df.groupby('case:concept:name')['concept:name'].shift(2).fillna('absent')
        train_df['concept:name - lag_3'] = train_df.groupby('case:concept:name')['concept:name'].shift(3).fillna('absent')

        # define target
        train_df['next concept:name'] = train_df.groupby('case:concept:name')['concept:name'].shift(-1).fillna('absent')

        
        test_df['concept:name - lag_1'] = test_df.groupby('case:concept:name')['concept:name'].shift(1).fillna('absent')
        test_df['concept:name - lag_2'] = test_df.groupby('case:concept:name')['concept:name'].shift(2).fillna('absent')
        test_df['concept:name - lag_3'] = test_df.groupby('case:concept:name')['concept:name'].shift(3).fillna('absent')

        # define target
        test_df['next concept:name'] = test_df.groupby('case:concept:name')['concept:name'].shift(-1).fillna('absent')

    return train_df, test_df