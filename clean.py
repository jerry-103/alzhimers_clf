import pandas as pd
from sklearn.model_selection import train_test_split

def clean(df, drop_cols):
    """
    :param df: Data frame to clean
    :param drop_cols: columns to drop from df
    :return: dataframe with cols dropped
    """
    cleaned_df = df.drop(labels = drop_cols, axis = 1)
    return cleaned_df

def split_data(df, target_var, test_size, random_seed = 42):
    """
    :param df: Dataframe to be split
    :param target_var: col_name of y_var to be predicted
    :param test_size: size of test set
    :param random_seed: seed for 'random_state' param
    :return: X & y train and test dataframes
    """
    #getting input (X) and target (y) features
    X = df.drop(target_var, axis = 1)
    y = df[target_var]
    #Splitting data using sklearn train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state= random_seed)
    return X_train, X_test, y_train, y_test