
#This file contains functions that converts csv-> pandas DF
import pandas as pd
def csv_to_df(path):
    """
    :param path: file path of csv
    :return: pandas dataframe, converted from csv
    """
    df = pd.read_csv(path)
    return df

