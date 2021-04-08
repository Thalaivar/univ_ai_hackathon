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
from xgboost import XGBClassifier

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
    model.fit(X, y, sample_weight=sample_wts)
    return model

def eval_model(model, df_train):
    X, y = df_train.drop(["risk_flag", "Id"], axis=1).values, df_train["risk_flag"].values
    scores = cross_val_score(model, X, y, cv=CV, n_jobs=1, scoring="roc_auc", error_score="raise", fit_params={"sample_weight": sample_wts})
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
    "n_estimators": 2000,
    "tree_method": "gpu_hist"
}
    
    
def train_stacking_model():
    base_models = [
            ("rfv1", RandomForestClassifier(n_jobs=2)),
            ("rfv2", RandomForestClassifier(class_weight="balanced", max_features=1, n_jobs=2)),
            ("rfv3", RandomForestClassifier(criterion="entropy", n_jobs=2, max_depth=10)),
            ("rfv4", RandomForestClassifier(criterion="entropy", n_jobs=2, max_features=2, max_depth=15)),
            ("xgb", XGBClassifier(n_estimators=1000, learning_rate=0.1, tree_method="gpu_hist", scale_pos_weight=7.13, min_child_weight=2, colsample_bytree=0.9)),
            ("extratreesv2", ExtraTreesClassifier(n_estimators=200, n_jobs=2, class_weight="balanced", max_depth=10)),
            ("extratreesv3", ExtraTreesClassifier(n_estimators=200, n_jobs=2, class_weight="balanced", criterion="entropy", max_features=1, max_depth=10)),
            ("adb", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=200, learning_rate=0.1))
        ]
    final_clf = RandomForestClassifier(n_jobs=2, class_weight="balanced")
    model = StackingClassifier(estimators=base_models, final_estimator=final_clf, cv=CV, passthrough=True)
    

if __name__ == "__main__":
    df_train, df_test = data_preprocess(
        drop_cols=["house_ownership", "car_ownership", "married"],
        single_transforms=SINGLE_TRANSFORMS,
        double_transforms=DOUBLE_TRANSFORMS
    )

    model = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_features=1, n_estimators=1000)
    # model = XGBClassifier(
    #         n_estimators=5000, 
    #         learning_rate=0.01, 
    #         tree_method="gpu_hist", 
    #         scale_pos_weight=12, 
    #         min_child_weight=2, 
    #         colsample_bytree=0.6, 
    #         subsample=0.9,
    #         gamma=0.5
    #     )

    print(eval_model(model, df_train))

    make_submission(
            model,
            df_train,
            df_test,
            "./tuning_rf/sub5.csv"
        )

    # print(compare_submissions("./goodsubmits/submission.csv", "./tuning_rf/sub5.csv"))