import pandas as pd
import numpy as np
import scipy.stats as ss

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from transforms import *

import warnings
warnings.filterwarnings('ignore')

TRAIN_FILE = "./train.csv"
TEST_FILE = "./test.csv"
N_FOLDS = 5
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

SINGLE_TRANSFORMS = [
    (scale_by_col, {"cols": ["current_job_years"], "scale_cols": ["current_house_years"]}),
    (scale_by_group, {"cols": ["income", "income"], "groups": ["city", "profession"]}),
    (add_house_status, {}),
    (custom_transform1, {})
]

TRANSFORMS_2 = [
    (scale_by_group, {"cols": ["age", "current_job_years", "current_house_years"], "groups": ["profession", "profession", "house_ownership"]}),
]
def data_preprocess(drop_cols=[], transforms=[]):
    df_train, df_test = (
            pd.read_csv(TRAIN_FILE), 
            pd.read_csv(TEST_FILE)
        )
    
    df_train = correct_names(df_train, cols=["profession", "city", "state"])
    df_test = correct_names(df_test, cols=["profession", "city", "state"])

    catg_cols = ["house_ownership", "car_ownership", "married", "profession", "city", "state"]
    df_train, df_test = convert_categorical(df_train, df_test, catg_cols)

    for (fn, kwargs) in transforms:
        df_train, df_test = fn(df_train, **kwargs), fn(df_test, **kwargs)

    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)


    # print(f"Training data shape: {df_train.shape} ; Test data shape: {df_test.shape}")
    return df_train, df_test

def train_model(model, df_train, fit_params={}):
    X, y = df_train.drop(["risk_flag", "Id"], axis=1).values, df_train["risk_flag"].values
    model.fit(X, y)
    return model

def eval_model(model, df_train):
    X, y = df_train.drop(["risk_flag", "Id"], axis=1).values, df_train["risk_flag"].values
    scores = cross_val_score(model, X, y, cv=CV, n_jobs=1, scoring="roc_auc", error_score="raise")
    return scores

def make_submission(model, df_train, df_test, filename):
    model = train_model(model, df_train)
    preds = model.predict(df_test.drop("id", axis=1).values)

    res = pd.DataFrame.from_dict({"id": np.arange(preds.size), "risk_flag": list(preds)})
    res.to_csv(filename, index=False)


def plot_rf_feats_imp(model, df_train):
    features = df_train.drop("risk_flag", axis=1).columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def compare_submissions(true, preds):
    true, preds = pd.read_csv(true), pd.read_csv(preds)
    print(roc_auc_score(true["risk_flag"].values, preds["risk_flag"].values))

RF_PARAMS = {
    "n_estimators": 1000,
    "n_jobs": -1,
    "max_features": 1,
    "class_weight": "balanced"
}

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "tree_method": "gpu_hist",
    "colsample_bytree": 0.9,
    "gamma": 0.1,
    "max_depth": 25,
    "scale_pos_weight": 9.267,
    "min_child_weight": 14.7,
    "subsample": 0.9
}

LGBM_PARAMS = {
    "device": "gpu",
    "boosting_type": "goss",
    "verbose": -1,
    "colsample_bytree": 0.7112,
    "max_bin": 191,
    "max_depth": 25,
    "num_leaves": 299,
    "scale_pos_weight": 12.231,
    "subsample": 0.239,
    "n_estimators": 1000,
    "learning_rate": 0.01
}

if __name__ == "__main__":
    df_train, df_test = data_preprocess(
        drop_cols=["house_ownership", "car_ownership", "married"],
        transforms=SINGLE_TRANSFORMS
    )

    
    opt_params = bayes_parameter_opt_lgb(df_train.drop("Id", axis=1).values, df_train["risk_flag"].values, init_round=25, opt_round=50, n_folds=3, random_seed=6,n_estimators=200)
    
    # model = LGBMClassifier(is_unbalance=True, n_estimators=200, learning_rate=0.1, device="gpu", num_leaves=200, boosting_type="goss", subsample=0.6)
    # print(eval_model(model, df_train))


    # make_submission(
    #         model,
    #         df_train,
    #         df_test,
    #         "./tuning_rf/sub5.csv"
    #     )

    # print(compare_submissions("./goodsubmits/submission.csv", "./tuning_rf/sub5.csv"))