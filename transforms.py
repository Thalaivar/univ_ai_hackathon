import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

def scale_by_group(df, cols, groups):
    for col, group in zip(cols, groups):
        df[f"{col}_per_{group}"] = df.groupby(group)[col].transform("median")
    return df

def subtract_from_group(df, cols, groups):
    for (col, group) in zip(cols, groups):
        df[col] = df[col] - df.groupby(group)[col].transform("median")
    return df
    
def make_biased_dataset(df):
    if "risk_flag" in df.columns:
        y = df["risk_flag"].values
        psr = y.sum() / (y.size - y.sum())
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.random.permutation(np.arange(y.size))[:int(psr * pos_idx.size)]
        idx = np.random.permutation(np.hstack((pos_idx, neg_idx)))
        df = df.loc[idx, :]
        return df
    else:
        return df    

def subtract_cols(train_df, test_df, cols):
    col1, col2 = cols
    train_df[f"{col1}-{col2}"] = train_df[col1] - train_df[col2]
    test_df[f"{col1}-{col2}"] = test_df[col1] - test_df[col2]
    return train_df, test_df

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
    df["has_house"] = df["house_ownership"].apply(lambda s: int(s != "norent_noown"))
    return df 

def add_job_status(df: pd.DataFrame) -> pd.DataFrame:
    def has_job(years: int):
        if years > 0:
            return 1
        else:
            return 0
    df["has_job"] = df["current_job_years"].apply(has_job)
    return df

def custom_transform1(df):
    df["common_house_car_married"] = df["has_house"] & df["car_ownership"] & df["married"]
    return df

# def defaults_per_col(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: str):
#     for col in cols:
#         df = train_df.groupby(col)["risk_flag"]
#         train_df[f"default_per_{col}"] = df.transform("sum")
#         colwise = df.sum()
#         colwise.name = f"default_per_{col}"
#         test_df = test_df.merge(colwise, how="right", on=col)
#     return train_df, test_df

def count_col_by_group(df, cols, groups):
    for col, group in zip(cols, groups):
        df[f"{col}_count"] = df.groupby(group)[col].transform("sum")
    return df