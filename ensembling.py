from main import data_preprocess, SINGLE_TRANSFORMS

from sklearn import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier, 
        ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ORIGNAL_SIZE = 252000

from main import *

"""
*********************************************************************
*   IF YOU CHANGE ANY OF THE CONFIGURATIONS FOR AN EXISTING MODEL,  *
*   DELETE THE SAVE FILES IN "./ensembles_files"                    *
*********************************************************************
"""
model_list = [
    ("rf", RandomForestClassifier(), RF_PARAMS, False),
    ("lgbm", LGBMClassifier(), LGBM_PARAMS, False),
    ("xgb", XGBClassifier(), XGB_PARAMS, False),
    ("lgbm-v2", LGBMClassifier(), LGBM_PARAMS, False),
    ("rf-v2", XGBClassifier(), XGB_PARAMS, False)
] 

transform_list = {
    "rf": (["house_ownership", "car_ownership", "married"], []),
    "lgbm": ([], SINGLE_TRANSFORMS),
    "xgb": (["house_ownership", "car_ownership", "married"], SINGLE_TRANSFORMS),
    "lgbm-v2": ([], TRANSFORMS_2),
    "rf-v2": ([], TRANSFORMS_3)
}

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

CV = StratifiedKFold(n_splits=5, shuffle=True)

def base_level_predictions(model, drop_cols=[], transforms=[]):
    df_train, df_test = data_preprocess(drop_cols, transforms)

    ids, y = df_train["Id"].values, df_train["risk_flag"].values
    X = df_train.drop(["risk_flag", "Id"], axis=1).values

    meta_train = {"id": [], "preds": [], "targets": []}
    for (train_idx, test_idx) in CV.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        
        meta_train["preds"].extend(model.predict_proba(X[test_idx])[:,0])
        meta_train["id"].extend(ids[test_idx])
        meta_train["targets"].extend(y[test_idx])
    
    meta_train = pd.DataFrame.from_dict(meta_train).sort_values(by="id")

    model.fit(X, y)
    meta_test = model.predict_proba(df_test.drop("id", axis=1).values)[:,0]
    meta_test = pd.DataFrame.from_dict({"id": np.arange(meta_test.size), "preds": meta_test})
    return (meta_train, meta_test)

def biased_base_level_predictions(model, drop_cols=[], transforms=[]):
    df_train, df_test = data_preprocess(drop_cols, transforms)
    df_bias = make_biased_dataset(df_train.copy())

    X = df_bias.drop(["risk_flag", "Id"], axis=1).values
    y = df_bias["risk_flag"].values
    model.fit(X, y)

    meta_train = {}
    meta_train["id"] = df_train["Id"].values
    meta_train["preds"] = model.predict_proba(df_train.drop(["Id", "risk_flag"], axis=1).values)[:,0]
    meta_train["targets"] = df_train["risk_flag"].values

    meta_train = pd.DataFrame.from_dict(meta_train)
    meta_test = model.predict_proba(df_test.drop("id", axis=1).values)[:,0]
    meta_test = pd.DataFrame.from_dict({"id": np.arange(meta_test.size), "preds": meta_test})
    return (meta_train, meta_test)
    
def generate_meta_datasets(model_list, transform_list):
    X_meta, X_test = [], []
    for i in tqdm(range(len(model_list))):
        name, model, model_params, use_bias = model_list[i]
        model = clone(model)
        model.set_params(**model_params)

        meta_train_fname = f"./ensemble_files/{name}_meta_train.csv"
        meta_test_fname = f"./ensemble_files/{name}_meta_test.csv"
        if os.path.exists(meta_train_fname) and os.path.exists(meta_test_fname):
            X_meta.append(pd.read_csv(meta_train_fname))
            X_test.append(pd.read_csv(meta_test_fname))
        else:
            drop_cols, transforms = transform_list[name]
            if use_bias:
                meta_train, meta_test = biased_base_level_predictions(model, drop_cols, transforms)
            else:
                meta_train, meta_test = base_level_predictions(model, drop_cols, transforms)

            meta_train.rename(columns={"preds": f"{name}_preds"}, inplace=True)
            meta_test.rename(columns={"preds": f"{name}_preds"}, inplace=True)
            meta_train.to_csv(meta_train_fname, index=False)
            meta_test.to_csv(meta_test_fname, index=False)

            X_meta, X_test = X_meta + [meta_train], X_test + [meta_test]

    df_train, targets = X_meta[0].drop("targets", axis=1), X_meta[0]["targets"].values
    for df in X_meta[1:-1]:
        assert not np.any(targets != df["targets"].values)
        df_train = df_train.merge(df.drop("targets", axis=1), on="id")
    df_train = df_train.merge(X_meta[-1], on="id")

    df_test = X_test[0]
    for df in X_test[1:]:
        df_test = df_test.merge(df, on="id")
    
    df_train.to_csv("./ensemble_files/final_train.csv", index=False)
    df_test.to_csv("./ensemble_files/final_test.csv", index=False)

    return df_train, df_test

def make_ensemble_submission(model):
    df_train, df_test = pd.read_csv("./ensemble_files/final_train.csv"), pd.read_csv("./ensemble_files/final_test.csv")
    
    X, y = df_train.drop(["targets", "id"], axis=1).values, df_train["targets"]
    model.fit(X, y)

    preds = model.predict(df_test.drop("id", axis=1).values)
    
    sub = pd.DataFrame.from_dict({"id": df_test["id"], "risk_flag": preds})
    sub.to_csv("./ensemble_files/stacking_sub.csv", index=False)

def max_voting_submission(df_test):
    preds = df_test.drop("id", axis=1).mean(axis=1)
    preds = preds.apply(lambda x: int(not x > 0.5))
    sub = pd.DataFrame.from_dict({"id": df_test["id"], "risk_flag": preds})
    sub.to_csv("./ensemble_files/maxvoting_sub.csv", index=False)

if __name__ == "__main__":
    df_train, df_test = generate_meta_datasets(model_list, transform_list)
    # model = XGBClassifier(gamma=0.5, tree_method="gpu_hist", n_estimators=5000, learning_rate=0.01, subsample=0.6)
    # # RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight="balanced")
    # make_submission(model)

    from sklearn.svm import LinearSVC
    
    make_ensemble_submission(RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, max_features=1))
    # model = LGBMClassifier(colsample_bytree=0.8565, max_depth=25, max_bin=180, num_leaves=188, scale_pos_weight=12.21, subsample=0.44, boosting_type="goss", device="gpu", n_estimators=2000)
    # make_ensemble_submission(model)
    # max_voting_submission(df_test)

    from main import compare_submissions
    print(compare_submissions("./goodsubmits/stacking_sub2.csv", "./ensemble_files/stacking_sub.csv"))