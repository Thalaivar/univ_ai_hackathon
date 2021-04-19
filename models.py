from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from transforms import *

from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline

def data_preprocess(dtrain, dtest=None, drop_cols=[], no_encode=False):
    df_train = dtrain.copy()

    if dtest is not None:
        df_test = dtest.copy()
    else:
        df_test = None
        
    df_train = df_train.rename(columns={"Id": "id"})
    df_train = correct_names(df_train, cols=["profession", "city", "state"])
    if df_test is not None:
        df_test = correct_names(df_test, cols=["profession", "city", "state"])

    if not no_encode:
        catg_cols = ["house_ownership", "car_ownership",
                     "married", "profession", "city", "state"]

        df_train = convert_categorical(df_train, df_test, cols=catg_cols)
        if df_test is not None:
            df_train, df_test = df_train

    df_train = df_train.drop(drop_cols, axis=1)
    if df_test is not None:
        df_test = df_test.drop(drop_cols, axis=1)
        return df_train, df_test
    return df_train

def rfv1_transform_fn(df, target_name="risk_flag"):
    over = SMOTE(sampling_strategy=0.75, k_neighbors=5)
    under = InstanceHardnessThreshold(sampling_strategy=1, n_jobs=-1)
    sampler = Pipeline(steps=[('o', over), ('u', under)])

    X, y = df.drop(["id", target_name], axis=1), df[target_name].values
    X, y = sampler.fit_resample(X, y)

    return X, y

def xgbv1_transform_fn(df, target_name="risk_flag"):
    X, y = df.drop(["id", target_name], axis=1), df[target_name].values
    X, y = AllKNN(n_jobs=-1, n_neighbors=20).fit_resample(X, y)
    return X, y

def lgbv1_transform_fn(df, target_name="risk_flag"):
    X, y = df.drop(["id", target_name], axis=1), df[target_name].values
    X, y = SMOTENC(sampling_strategy=1, categorical_features=[3, 4, 5, 6, 7, 8], n_jobs=-1).fit_resample(X, y)
    return X, y

base_models = {
    "rfv1": {
        "model": RandomForestClassifier(
                    n_estimators=500,
                    n_jobs=-1,
                    max_features=1
                ),
        "transform_fn": rfv1_transform_fn,
        "preprocess_fn": data_preprocess
    },

    "xgbv1": {
        "model": XGBClassifier(
                    tree_method="gpu_hist",
                    n_estimators=1000,
                    eval_metric="logloss",
                    learning_rate=0.1,
                    max_depth=23,
                    min_child_weight=3.481,
                    gamma=0,
                    reg_alpha=10 ** -3.4825
                ),
        "transform_fn": xgbv1_transform_fn,
        "preprocess_fn": data_preprocess 
    },

    "lgbv1": {
        "model": LGBMClassifier(
                    learning_rate=0.1,
                    n_estimators=200,
                    device="gpu",
                    num_leaves=2 ** 16,
                    boosting_type="goss",
                    max_depth=23,
                    max_bin=64,
                    min_child_weight=1.899,
                    lambda_l1=10 ** -3.565,
                    lambda_l2=10 ** -2.953,
                    verbose=-1
                ),
        "transform_fn": lgbv1_transform_fn,
        "preprocess_fn": data_preprocess
    }
}

meta_learner = {
    "model": RandomForestClassifier(max_depth=5, max_features=1, n_estimators=1000, n_jobs=-1, class_weight="balanced"),
    "transform_fn": None,
    "preprocess_fn": data_preprocess
}

MODEL_LIST = {
    "xgb": {
        "model": XGBClassifier(),
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "tree_method": "gpu_hist",
            "gamma": 0.938358,
            "max_depth": 8,
            # "scale_pos_weight": 19.35097,
            "min_child_weight": 4.525,
        },
        "fit_params": {}
    },

    "lgb": {
        "model": LGBMClassifier(),
        "params": {
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "device": "gpu",
            "boosting_type": "goss",
            "num_leaves": 229,
            # "scale_pos_weight": 13.56929,
            "colsample_bytree": 0.95781,
            "subsample": 0.74331,
            "max_depth": 7,
            "min_child_weight": 7.5546,
            "max_bin": 96,
            "lambda_l1": 0.0868,
            "lambda_l2": 0.01541,
            "verbose": -1
        },
        "fit_params": {}
    },

    "cboost": {
        "model": CatBoostClassifier(),
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "loss_function": "Logloss",
            "boosting_type": "Ordered",
            "eval_metric": "AUC",
            "cat_features":  ["house_ownership", "car_ownership", "married",
                               "city", "profession", "state", "age", "experience", "income"],
            "max_depth": 5,
            "task_type": "GPU",
            "auto_class_weights": "Balanced",
            "verbose": False
        },
        "fit_params": {}
    },

    "super": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": 1000,
            "n_jobs": -1,
            "max_depth": 5,
            "max_features": 1
        },
        "fit_params": {}
    }
}

TRANSFORM_LIST = {
    "xgb": {
        "drop_cols": [],
        "transforms": [],
        "no_encode": False
    },

    "lgb": {
        "drop_cols": [],
        "transforms": [],
        "no_encode": False
    },

    "cboost": {
        "drop_cols": [],
        "transforms": [(catboost_dataset, {"cols": ["house_ownership", "car_ownership", "married",
                                                    "city", "profession", "state", "age", "experience", "income"], })],
        "no_encode": True
    },
}
