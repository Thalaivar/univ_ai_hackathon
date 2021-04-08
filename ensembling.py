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

rf = RandomForestClassifier(n_jobs=-1)
dtree = DecisionTreeClassifier()
etree = ExtraTreesClassifier(n_jobs=-1)

"""
*********************************************************************
*   IF YOU CHANGE ANY OF THE CONFIGURATIONS FOR AN EXISTING MODEL,  *
*   DELETE THE SAVE FILES IN "./ensembles_files"                    *
*********************************************************************
"""
model_list = [
    ("rfv1", rf, {"n_estimators": 500, "class_weight": "balanced", "max_features": 1}, {}),
    ("rfv2", rf, {"n_estimators": 500, "max_features": 0.5}, {}),
    ("rfv3", rf, {"class_weight": "balanced", "criterion": "entropy", "max_features": 0.8}, {}),
]
transform_list = {
    "rfv1": (["house_ownership", "car_ownership", "married"], []),
    "rfv2": ([], SINGLE_TRANSFORMS),
    "rfv3": (["income", "age", "current_job_years", "current_house_years"], [])
}

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

CV = StratifiedKFold(n_splits=10, shuffle=True)

def base_level_predictions(model, drop_cols=[], transforms=[], fit_params={}):
    df_train, df_test = data_preprocess(drop_cols, transforms)
    ids, y = df_train["Id"].values, df_train["risk_flag"].values
    X = df_train.drop(["risk_flag", "Id"], axis=1).values

    meta_train = {"id": [], "preds": [], "targets": []}
    for (train_idx, test_idx) in CV.split(X, y):
        model.fit(X[train_idx], y[train_idx], **fit_params)
        
        meta_train["preds"].extend(model.predict(X[test_idx]))
        meta_train["id"].extend(ids[test_idx])
        meta_train["targets"].extend(y[test_idx])
    
    meta_train = pd.DataFrame.from_dict(meta_train).sort_values(by="id")

    model.fit(X, y)
    meta_test = model.predict_proba(df_test.drop("id", axis=1).values)[:,0]
    meta_test = pd.DataFrame.from_dict({"id": np.arange(meta_test.size), "preds": meta_test})
    return (meta_train, meta_test)

def generate_meta_datasets(model_list, transform_list):
    X_meta, X_test = [], []
    for i in tqdm(range(len(model_list))):
        name, model, model_params, fit_params = model_list[i]
        model = clone(model)
        model.set_params(**model_params)

        meta_train_fname = f"./ensemble_files/{name}_meta_train.csv"
        meta_test_fname = f"./ensemble_files/{name}_meta_test.csv"
        if os.path.exists(meta_train_fname) and os.path.exists(meta_test_fname):
            X_meta.append(pd.read_csv(meta_train_fname))
            X_test.append(pd.read_csv(meta_test_fname))
        else:
            drop_cols, transforms = transform_list[name]
            meta_train, meta_test = base_level_predictions(model, drop_cols, transforms, fit_params)

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

def make_submission(model):
    df_train, df_test = pd.read_csv("./ensemble_files/final_train.csv"), pd.read_csv("./ensemble_files/final_test.csv")
    
    X, y = df_train.drop(["targets", "id"], axis=1).values, df_train["targets"]
    model.fit(X, y)

    preds = model.predict(df_test.drop("id", axis=1).values)
    
    sub = pd.DataFrame.from_dict({"id": df_test["id"], "risk_flag": preds})
    sub.to_csv("./ensemble_files/stacking_sub.csv", index=False)

if __name__ == "__main__":
    df_train, df_test = generate_meta_datasets(model_list, transform_list)
    make_submission(RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight="balanced"))

    from main import compare_submissions
    print(compare_submissions("./goodsubmits/sub3.csv", "./ensemble_files/stacking_sub.csv"))