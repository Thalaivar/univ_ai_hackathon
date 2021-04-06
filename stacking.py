import pandas as pd
import numpy as np
import joblib
import scipy.stats as ss

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from itertools import combinations
from sklearn import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.DEBUG)

TRAIN_FILE = "./train.csv"
TEST_FILE = "./test.csv"
N_FOLDS = 10
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

from sklearn.metrics import roc_auc_score

def datapreprocess(train_only=False):
    df_train, df_test = (
            pd.read_csv(TRAIN_FILE).drop("Id", axis=1), 
            pd.read_csv(TEST_FILE).drop("id", axis=1)
        )

    for col in ["profession", "city", "state"]:
        df_train[col] = df_train[col].str.replace("_", " ")
        df_test[col] = df_test[col].str.replace("_", " ")
    
    def smoothed_target_encode(df, m, overall_median):
        w = df.shape[0] / (df.shape[0] + m)
        return w * df.median() + (1 - w) * overall_median
        
    for col in ["city", "state"]:
        transform_args = {"m": 2500, "overall_median": df_train["income"].median()}
        df_train["median_" + col + "_income"] = df_train.groupby(col)["income"].transform(smoothed_target_encode, **transform_args)
        transform_args = {"m": 250, "overall_median": df_test["income"].median()}
        df_test["median_" + col + "_income"] = df_test.groupby(col)["income"].transform(smoothed_target_encode, **transform_args)

    encode_cols = ["house_ownership", "car_ownership", "married", "profession", "city", "state"]
    for col in encode_cols:
        encoder = LabelEncoder().fit(df_train[col].values)
        df_train[col] = encoder.transform(df_train[col].values)
        df_test[col] = encoder.transform(df_test[col].values)
    
    for df in [df_train, df_test]:
        df["income_for_profession"] = df.groupby("profession")["income"].transform("median")
        df["income_for_profession"] = df["income"] / df["income_for_profession"]
        
    df_train["lifescore"] = df_train[["married", "house_ownership", "car_ownership"]].sum(axis=1)
    df_test["lifescore"] = df_test[["married", "house_ownership", "car_ownership"]].sum(axis=1)
    
    drop_cols = ["car_ownership", "house_ownership", "married", "profession", "city", "state", "profession"]
    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)
    
    if train_only:
        return df_train
    return df_train, df_test
    
def make_submission(clf):
    df_train, df_test = datapreprocess()
    X, y = df_train.drop("risk_flag", axis=1).values, df_train["risk_flag"].values
    clf.fit(X, y)
    
    preds = clf.predict(df_test.values)
    res = {"id": np.arange(preds.size), "risk_flag": list(preds)}
    return pd.DataFrame.from_dict(res)

def train(clf, df_train=None):
    if df_train is None:
        df_train = datapreprocess(train_only=True)
    X, y = df_train.drop("risk_flag", axis=1).values, df_train["risk_flag"].values
    clf.fit(X, y)
    return clf, roc_auc_score(y, clf.predict(X))

def eval_model(clf, df_train=None, n_jobs=-1):
    if df_train is None:
        df_train = datapreprocess(train_only=True)
    X, y = df_train.drop("risk_flag", axis=1).values, df_train["risk_flag"].values
    scores = cross_val_score(clf, X, y, cv=CV, n_jobs=n_jobs, scoring="roc_auc")
    print(f"AUC: {scores.mean()} +- {scores.std()}")
    
from joblib import Parallel, delayed
class BiasedStackedClassifier:
    def __init__(self, base_classifiers, final_classifier, n_jobs=-1):
        self.base_clfs = [clone(clf) for clf in base_classifiers]
        self.biased_clfs = [clone(clf) for clf in base_classifiers]
        self.final_clf = clone(final_classifier)
        self.n_jobs = n_jobs
    
    def make_biased_dataset(self, X, y, psr=None):
        if psr is None:
            psr = (y.size - y.sum()) / y.sum()

        pos_idx = np.where(y == 1)[0]
        n_neg = int(pos_idx.size / psr)
        neg_idx = np.random.permutation(np.where(y == 0)[0])[:n_neg]

        idx = np.random.permutation(np.concatenate((pos_idx, neg_idx), axis=0))
        return X[idx], y[idx]

    def fit_single_classifier(self, clf, X, y):
        logging.debug(f"training: {str(clf)} on {X.shape} data")
        clf.fit(X, y)
        return clf

    def fit_base_classifiers(self, X, y):
        self.base_clfs = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.fit_single_classifier)(clone(clf), X, y) 
                        for clf in self.base_clfs
                    )

        logging.debug("training biased models")
        X, y = self.make_biased_dataset(X, y)
        self.biased_clfs = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.fit_single_classifier)(clone(clf), X, y)
                        for clf in self.biased_clfs
                    )

    def base_clf_predict(self, X):
        preds = [clf.predict(X) for clf in self.base_clfs]
        preds.extend([clf.predict(X) for clf in self.biased_clfs])

        return np.hstack([p.reshape(-1,1) for p in preds])

    def fit_final_classifer(self, X, y):
        X_meta, y_meta = [], []
        logging.debug(f"generating data set for final classifier: {str(self.final_clf)}")
        for i, (train_idx, test_idx) in enumerate(CV.split(X, y)):
            self.fit_base_classifiers(X[train_idx], y[train_idx])
            X_meta.append(self.base_clf_predict(X[test_idx]))
            y_meta.append(y[test_idx])
            logging.debug(f"generated {i+1}/{N_FOLDS} splits")
        
        X_meta, y_meta = np.vstack(X_meta), np.hstack(y_meta)
        logging.debug(f"training final classifier on {X_meta.shape} data")
        self.final_clf = self.fit_single_classifier(clone(self.final_clf), X_meta, y_meta)
    
    def fit(self, X, y):
        self.fit_final_classifer(X, y)
        self.fit_base_classifiers(X, y)
    
    def predict(self, X):
        X_meta = self.base_clf_predict(X)
        return self.final_clf.predict(X_meta)
    
    def cv_each_base(self, X, y, cv=CV):
        results = {}
        for clf in self.base_clfs:
            scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
            results[str(clf)] = (scores.mean(), scores.std())
        
        for clf in self.biased_clfs:
            scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
            results[f"bias_{str(clf)}"] = (scores.mean(), scores.std())
        
        return results
    
    def cv_final_clf(self, X, y, cv=CV):
        X_meta, y_meta = [], []
        logging.debug(f"generating data set for final classifier: {str(self.final_clf)}")
        for i, (train_idx, test_idx) in enumerate(CV.split(X, y)):
            self.fit_base_classifiers(X[train_idx], y[train_idx])
            X_meta.append(self.base_clf_predict(X[test_idx]))
            y_meta.append(y[test_idx])
            logging.debug(f"generated {i+1}/{N_FOLDS} splits")
        
        X_meta, y_meta = np.vstack(X_meta), np.hstack(y_meta)
        scores = cross_val_score(self.final_clf, X_meta, y_meta, scoring="roc_auc", cv=cv)        
        return (scores.mean(), scores.std())
    
    def cv(self, X, y, cv=CV):
        results = self.cv_each_base(X, y, cv)
        results["final_clf"] = self.cv_final_clf(X, y, cv)
        return results

    def save_models(self):
        with open("./stacked.model", "rb") as f:
            joblib.dump([self.base_clfs, self.biased_clfs, self.final_clf], f)
    
    def load_models(self):
        with open("./stacked.model", "rb") as f:
            self.base_clfs, self.biased_clfs, self.final_clf = joblib.load(f)