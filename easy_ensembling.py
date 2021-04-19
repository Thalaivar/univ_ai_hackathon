import os
import sklearn
import pandas as pd
import numpy as np

from sklearn import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.combine import 

def evaluate_super_learner(save_dir: str, models: dict, cv=5, metric="roc_auc"):
    df_train = pd.read_csv(os.path.join(save_dir, "ensemble_train.csv"))
    model = clone(models["super"]["model"])
    model.set_params(**models["super"]["params"])

    X, y = df_train.drop("targets", axis=1).values, df_train["targets"].values
    results = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=1,
                              error_score="raise", fit_params=models["super"]["fit_params"])

    return results


def evaluate_base_model(data_fn, targets: str, model_data: dict, transforms: dict, cv=5, metric="roc_auc"):
    model = clone(model_data["model"])
    model.set_params(**model_data["params"])

    df_train, _ = data_fn(**transforms)
    X, y = df_train.drop(targets, axis=1).values, df_train[targets].values

    results = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=1,
                              error_score="raise", fit_params=model_data["fit_params"])

    return results


def evaluate_all_base_models(data_fn, targets: str, models: dict, transforms: dict, cv=5, metric="roc_auc"):
    results = {}
    for name, model_data in models.items():
        print(f"Cross-validating model: {name}")
        results[name] = evaluate_base_model(
            data_fn, targets, model_data, transforms[name], cv, metric)

    return results


def generate_ensemble_predictions(models: dict, save_dir: str, target_name="preds"):
    if not os.path.exists(os.path.join(save_dir, "ensemble_train.csv")) or not os.path.exists(os.path.join(save_dir, "ensemble_test.csv")):
        raise FileNotFoundError

    df_train = pd.read_csv(os.path.join(save_dir, "ensemble_train.csv"))
    df_test = pd.read_csv(os.path.join(save_dir, "ensemble_test.csv"))

    X, y = df_train.drop(["targets", "id"], axis=1), df_train["targets"]

    model = clone(models["super"]["model"])
    model.set_params(**models["super"]["params"])
    model.fit(X, y, **models["super"]["fit_params"])

    X_test = df_test.drop("id", axis=1)
    preds = model.predict(X_test)

    sub = pd.DataFrame.from_dict({"id": df_test["id"], target_name: preds})
    sub.to_csv(os.path.join(save_dir, "stacking_sub.csv"), index=False)


def base_level_predictions(model, data_fn, targets, cv=5, data_fn_args={}, fit_params={}) -> (pd.DataFrame, pd.DataFrame):
    if type(cv) == int:
        cv = StratifiedKFold(n_splits=cv, shuffle=True)

    df_train, df_test = data_fn(**data_fn_args)

    drop_cols = [targets] + ["id"]
    X, y = df_train.drop(drop_cols, axis=1), df_train[targets].values

    
    meta_train = {"id": [], "preds": [], "targets": []}
    for (train_idx, test_idx) in cv.split(X, y):
        df = df_train.iloc[train_idx]
        X_train, y_train = df.drop(drop_cols, axis=1), df[targets].values
        model.fit(X_train, y_train, **fit_params)

        df = df_train.iloc[test_idx]
        X_test, y_test = df.drop(drop_cols, axis=1), df[targets].values

        meta_train["preds"].extend(model.predict_proba(X_test)[:, 0])
        meta_train["id"].extend(df["id"].tolist())
        meta_train["targets"].extend(y_test)

    meta_train = pd.DataFrame.from_dict(meta_train).sort_values(by="id")

    X, y = df_train.drop(drop_cols, axis=1), df_train[targets].values
    model.fit(X, y, **fit_params)

    meta_test = model.predict_proba(df_test.drop("id", axis=1).values)[:, 0]
    meta_test = pd.DataFrame.from_dict(
        {"id": df_test["id"].values, "preds": meta_test})

    return (meta_train, meta_test)


def generate_meta_datasets(data_fn, targets: str, models: dict, transforms: dict, save_dir: str, cv: int = 5, passthrough=False, pred_type="proba") -> (pd.DataFrame, pd.DataFrame):
    X_meta, X_test = [], []

    for name, model_data in models.items():
        if name != "super":
            meta_train_fname = os.path.join(save_dir, f"{name}_meta_train.csv")
            meta_test_fname = os.path.join(save_dir, f"{name}_meta_test.csv")

            if os.path.exists(meta_train_fname) and os.path.exists(meta_test_fname):
                X_meta.append(pd.read_csv(meta_train_fname))
                X_test.append(pd.read_csv(meta_test_fname))
            else:
                print(f"Generating meta datasets for model: {name}")
                model = clone(model_data["model"])
                model.set_params(**model_data["params"])

                meta_train, meta_test = base_level_predictions(
                    model,
                    data_fn,
                    targets,
                    cv=cv,
                    data_fn_args=transforms[name],
                    fit_params=model_data["fit_params"]
                )

                meta_train.rename(
                    columns={"preds": f"{name}_preds"}, inplace=True)
                meta_test.rename(
                    columns={"preds": f"{name}_preds"}, inplace=True)
                meta_train.to_csv(meta_train_fname, index=False)
                meta_test.to_csv(meta_test_fname, index=False)

                X_meta, X_test = X_meta + [meta_train], X_test + [meta_test]

    meta_train, labels = X_meta[0].drop(
        "targets", axis=1), X_meta[0]["targets"].values
    for df in X_meta[1:-1]:
        assert not np.any(labels != df["targets"].values)
        meta_train = meta_train.merge(df.drop("targets", axis=1), on="id")
    meta_train = meta_train.merge(X_meta[-1], on="id")

    meta_test = X_test[0]
    for df in X_test[1:]:
        meta_test = meta_test.merge(df, on="id")

    if pred_type == "label":
        meta_train, meta_test = convert_proba_to_pred(meta_train), convert_proba_to_pred(meta_test)

    if passthrough:
        df_train, df_test = data_fn(**transforms["super"])
        meta_train = meta_train.merge(df_train.drop(targets, axis=1), on="id")
        meta_test = meta_test.merge(df_test, on="id")

    meta_train.to_csv(os.path.join(
        save_dir, "ensemble_train.csv"), index=False)
    meta_test.to_csv(os.path.join(save_dir, "ensemble_test.csv"), index=False)

    return meta_train, meta_test

def convert_proba_to_pred(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "preds" in col:
            df[col] = df[col].apply(lambda x: int(not x >= 0.5))
            df[col] = df[col].astype("category")
    return df


def ensemble_learner(data_fn, targets: str, models: dict, transforms: dict, save_dir: str, cv=5, passthrough=False, pred_type="proba"):
    generate_meta_datasets(data_fn, targets, models,
                           transforms, save_dir, cv, passthrough, pred_type)
    generate_ensemble_predictions(models, save_dir, target_name=targets)


if __name__ == "__main__":
    from main import data_preprocess
    from models import MODEL_LIST, TRANSFORM_LIST
    ensemble_learner(data_preprocess, targets="risk_flag", models=MODEL_LIST,
                     transforms=TRANSFORM_LIST, save_dir="./ensemble_files", cv=5, passthrough=False, pred_type="proba")

    from main import compare_submissions
    print(compare_submissions("./goodsubmits/stacking_sub2.csv",
                              "./ensemble_files/stacking_sub.csv"))

    # scores = evaluate_super_learner(save_dir="./ensemble_files", models=MODEL_LIST)
    # print([scores.mean(), scores.std()])
