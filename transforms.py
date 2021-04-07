import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

def scaled_income(df: pd.DataFrame, col: str, name: str=None) -> pd.DataFrame:
    median_income = df.groupby(col)["income"].transform("median")
    if not name:
        df["income"] = df["income"] / median_income
    else:
        df[name] = df["income"] / median_income
    return df

def scale_cols(df: pd.DataFrame, col: str, scale_col: str, name: str=None) -> pd.DataFrame:
    name = col if not name else name
    df[name] = df[col] / df[scale_col]
    df[name] = df[name].replace([np.inf, -np.inf], 100)
    return df

def correct_names(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].str.replace("_", " ")
    return df

def convert_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    encoders = {} 
    for col in cols:
        encoder = LabelEncoder().fit(df[col].values)
        df[col] = encoder.transform(df[col].values)
        encoders[col] = encoder
    return df, encoders

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