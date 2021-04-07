import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

def scale_by_group(df, cols, groups):
    for col, group in zip(cols, groups):
        df[f"{col}_per_{group}"] = df.groupby(group)[col].transform("median")
    return df

def scale_by_col(df: pd.DataFrame, cols: str, scale_cols: str) -> pd.DataFrame:
    for col, scale_col in zip(cols, scale_cols):
        name = f"{col}_per_{scale_col}"
        df[name] = df[col] / df[scale_col]
        df[name] = df[name].replace([np.inf, -np.inf], df[col].max())
    return df

def correct_names(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].str.replace("_", " ")
    return df

def convert_categorical(train_df: pd.DataFrame, test_df:pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        encoder = LabelEncoder().fit(train_df[col].values)
        train_df[col] = encoder.transform(train_df[col].values)
        test_df[col] = encoder.transform(test_df[col].values)
    return train_df, test_df

def add_house_status(df: pd.DataFrame) -> pd.DataFrame:
    def has_house(s: str):
        if s == "norent_noown":
            return 0
        else:
            return 1
    df["has_house"] = df["house_ownership"].apply(has_house)
    return df 

def add_job_status(df: pd.DataFrame) -> pd.DataFrame:
    def has_job(years: int):
        if years > 0:
            return 1
        else:
            return 0
    df["has_job"] = df["current_job_years"].apply(has_job)
    return df

def defaults_per_col(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: str):
    for col in cols:
        df = train_df.groupby(col)["risk_flag"]
        train_df[f"default_per_{col}"] = df.transform(lambda df: df.sum()/df.shape[0])
        colwise = df.apply(lambda df: df.sum()/df.shape[0])
        colwise.name = f"default_per_{col}"
        test_df = test_df.merge(colwise, how="right", on=col)
    return train_df, test_df