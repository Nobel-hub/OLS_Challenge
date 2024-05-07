from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
def find_constant_columns(dataframe):
    constant = []
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 1:
            constant.append(column)
    return constant

def delete_constant_columns(dataframe, columns_to_delete):
    dataframe.drop(columns_to_delete, axis=1)
    return dataframe

def find_columns_with_fewer_values(dataframe, threshold):
    few_value_columns = []

    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) < threshold:
            few_value_columns.append(column)
    return few_value_columns

def find_duplicate_rows(dataframe):
    duplicated_rows = dataframe[dataframe.duplicated(keep='first')]
    return duplicated_rows

def find_nonnumber_columns(dataframe):
    nonnumber = []

    for column in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            nonnumber.append(column)
    return nonnumber

def drop_and_fill(data):
    cols_to_drop = data.columns[data.isnull().mean() > 0.5]
    data = data.drop(cols_to_drop, axis=1)
    data = data.fillna(data.mean())
    return data


def split_data(df, target_column, test_size=0.2, random_state=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


