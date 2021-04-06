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

import warnings
warnings.filterwarnings('ignore')

TRAIN_FILE = "./train.csv"
TEST_FILE = "./test.csv"
N_FOLDS = 10
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

def data_preprocess(drop_cols, transform_fns):
    df_train, df_test = (
            pd.read_csv(TRAIN_FILE).drop("Id", axis=1), 
            pd.read_csv(TEST_FILE).drop("id", axis=1)
        )

    for col in ["profession", "city", "state"]:
        df_train[col] = df_train[col].str.replace("_", " ")
        df_test[col] = df_test[col].str.replace("_", " ")

    encode_cols = ["house_ownership", "car_ownership", "married", "profession", "city", "state"]
    for col in encode_cols:
        encoder = LabelEncoder().fit(df_train[col].values)
        df_train[col] = encoder.transform(df_train[col].values)
        df_test[col] = encoder.transform(df_test[col].values)
    
    for fn in transform_fns:
        df_train, df_test = fn(df_train), fn(df_test)

    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)

    return df_train, df_test

def train_model(model, df_train):
    X, y = df_train.drop("risk_flag", axis=1).values, df_train["risk_flag"].values
    model.fit(X, y)
    return model

def eval_model(model, df_train):
    X, y = df_train.drop("risk_flag", axis=1).values, df_train["risk_flag"].values
    scores = cross_val_score(model, X, y, cv=CV, n_jobs=1, scoring="roc_auc", error_score="raise")
    return scores

def make_submission(model, df_train, df_test):
    model = train_model(model, df_train)
    preds = model.predict(df_test.values)

    res = {"id": np.arange(preds.size), "risk_flag": list(preds)}
    return pd.DataFrame.from_dict(res) 

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

def income_by_state(df):
    median_state_income = df.groupby("state")["income"].transform("median")
    df["income"] = df["income"] / median_state_income
    return df

def biased_data(df):
    y = df["risk_flag"].values
    psr = (y.size - y.sum()) / y.sum()
    
    pos_idx = np.where(y == 1)[0]
    # neg_idx = 
    
def train_stacking_model():
    base_models = [
            ("rfv1", RandomForestClassifier(n_jobs=2)),
            ("rfv2": RandomForestClassifier(class_weight="balanced", max_features=1, n_jobs=2)),
            ("rfv3": RandomForestClassifier(criterion="entropy", n_jobs=2, max_depth=10)),
            ("rfv4": RandomForestClassifier(criterion="entropy", n_jobs=2, max_features=2, max_depth=15)),
            ("xgb": XGBClassifier(n_estimators=1000, learning_rate=0.1, tree_method="gpu_hist", scale_pos_weight=7.13, min_child_weight=2, colsample_bytree=0.9)),
            ("extratreesv2": ExtraTreesClassifier(n_estimators=200, n_jobs=2, class_weight="balanced", max_depth=10)),
            ("extratreesv3": ExtraTreesClassifier(n_estimators=200, n_jobs=2, class_weight="balanced", criterion="entropy", max_features=1, max_depth=10)),
            ("adb": AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=200, learning_rate=0.1))
        ]
    final_clf = RandomForestClassifier(n_jobs=2, class_weight="balanced")
    model = StackingClassifier(estimators=base_models, final_estimator=final_clf, cv=CV, passthrough=True)

if __name__ == "__main__":
    df_train, df_test = data_preprocess(
        drop_cols=["car_ownership", "house_ownership", "married"],
        transform_fns=[income_by_state]
    )
    
    model = XGBClassifier(
            n_estimators=5000, 
            learning_rate=0.01,
            tree_method="gpu_hist",
            scale_pos_weight=7.13,
            min_child_weight=2,
            colsample_bytree=0.6,
            reg_alpha=1e-3,
        )

    sub = make_submission(model, df_train, df_test)
    sub.to_csv("./tuning_xgb/sub1.csv", index=False)