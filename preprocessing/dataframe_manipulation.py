import os
from typing import Any

import pandas as pd

from sklearn.model_selection import train_test_split


def get_dataframe(filename: str) -> pd.DataFrame:
    '''Creates a `pandas` DataFrame based on the given input filename, provided that the filename exist in the `data` folder'''
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")
    try:
        df = pd.read_csv(os.path.join(data_path, filename))
        return df
    except:
        print(f"Could not find file: {filename}")
        return None


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes the dataframe to the correct format
    '''
    df["sentiment"] = pd.cut(x=df["Rating"], bins=[0, 3, 6], labels=[0, 1])
    df = df.rename(columns={"Review": "text", "Rating": "rating"})
    return df


def get_and_process_df(filename: str) -> pd.DataFrame:
    '''Combines getting and processing the dataframe'''
    return process_dataframe(get_dataframe(filename))

def train_validate_test_split(X: Any, y: Any, test_size: float, val_size: float) -> Any:
    '''Creates train, validation and test split'''
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size)

    return X_train, X_val, X_test, y_train, y_val, y_test