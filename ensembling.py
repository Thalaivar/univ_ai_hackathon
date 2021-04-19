import os
import pandas as pd
import numpy as np

from sklearn import clone
from sklearn.model_selection import StratifiedKFold

CV = StratifiedKFold(n_splits=5, shuffle=True)
SAVE_DIR = "./ensemble_files"

class BaseModelPipeline:
    def __init__(self, model, preprocess_fn, transform_fn=None):
        self.model = model
        self.transform_fn = transform_fn
        self.preprocess_fn = preprocess_fn

    def get_train_features(self, train_file, target_name):
        dtrain = self.preprocess_fn(pd.read_csv(train_file))
        targets = dtrain[target_name].values
        
        meta_train = {"id": [], "preds": [], "targets": []}
        for (train_idx, test_idx) in CV.split(np.zeros_like(targets), targets):
            self.model = clone(self.model)
            X_train, X_test = dtrain.iloc[train_idx], dtrain.iloc[test_idx]

            if self.transform_fn:
                X_train, y_train = self.transform_fn(X_train, target_name)
            else:
                X_train, y_train = X_train.drop(["id", target_name], axis=1), X_train[target_name].values
            self.model.fit(X_train, y_train)
            
            meta_train["id"].extend(X_test["id"].tolist())
            meta_train["preds"].extend(self.model.predict_proba(X_test.drop(["id", target_name], axis=1))[:,0])
            meta_train["targets"].extend(X_test[target_name].tolist())
        
        meta_train = pd.DataFrame.from_dict(meta_train).sort_values(by="id")
        return meta_train   


    def get_test_features(self, train_file, test_file, target_name):
        dtrain, dtest = self.preprocess_fn(pd.read_csv(train_file), pd.read_csv(test_file))
        
        if self.transform_fn:
            X_train, y_train = self.transform_fn(dtrain, target_name)
        else:
            X_train, y_train = dtrain.drop(["id", "risk_flag"], axis=1), dtrain["risk_flag"].values
        
        self.model = clone(self.model)
        self.model.fit(X_train, y_train)
        preds = self.model.predict_proba(dtest.drop("id", axis=1))[:,0]

        meta_test = pd.DataFrame.from_dict({"id": dtest["id"], "preds": preds})
        return meta_test

class EnsembleLearner:
    def __init__(self, meta_learner, save_dir, preprocess_fn=None, transform_fn=None, **base_models):
        self.base_models = {name: BaseModelPipeline(**model_spec) for name, model_spec in base_models.items()}
        self.save_dir = save_dir
        self.preprocess_fn = preprocess_fn
        self.transform_fn = transform_fn
        self.meta_learner = meta_learner

    def generate_datasets(self, train_file, test_file, passthrough=False):
        X_meta, X_test = [], []
        
        for name, model in self.base_models.items():
            meta_train_fname = os.path.join(self.save_dir, f"{name}_meta_train.csv")
            meta_test_fname = os.path.join(self.save_dir, f"{name}_meta_test.csv")

            if os.path.exists(meta_train_fname) and os.path.exists(meta_test_fname):
                X_meta.append(pd.read_csv(meta_train_fname))
                X_test.append(pd.read_csv(meta_test_fname))
            else:
                print(f"Generating meta datasets for model: {name}")
                meta_train = model.get_train_features(train_file, target_name="risk_flag")
                meta_test = model.get_test_features(train_file, test_file, target_name="risk_flag")

                meta_train.rename(columns={"preds": f"{name}_preds"}, inplace=True)
                meta_train.to_csv(meta_train_fname, index=False)
                meta_test.rename(columns={"preds": f"{name}_preds"}, inplace=True)
                meta_test.to_csv(meta_test_fname, index=False)

                X_meta, X_test = X_meta + [meta_train], X_test + [meta_test]
        
        meta_train, labels = X_meta[0].drop("targets", axis=1), X_meta[0]["targets"].values
        for df in X_meta[1:-1]:
            assert not np.any(labels != df["targets"].values)
            meta_train = meta_train.merge(df.drop("targets", axis=1), on="id")
        meta_train = meta_train.merge(X_meta[-1], on="id")

        meta_test = X_test[0]
        for df in X_test[1:]:
            meta_test = meta_test.merge(df, on="id")

        if passthrough:
            if not self.preprocess_fn:
                raise ValueError("if using `passthrough` both `preprocess_fn` and `transform_fn` for the meta learner need to be specified")
            
            dtrain, dtest = self.preprocess_fn(pd.read_csv(train_file), pd.read_csv(test_file), target_name="risk_flag")

            meta_train = meta_train.merge(dtrain.drop("risk_flag", axis=1), on="id")
            meta_test = meta_test.merge(dtest, on="id")

        meta_train.to_csv(os.path.join(self.save_dir, "ensemble_train.csv"), index=False)
        meta_test.to_csv(os.path.join(self.save_dir, "ensemble_test.csv"), index=False)

        return meta_train, meta_test

    def predictions(self, train_file, test_file, passthrough=False):
        meta_train, meta_test = self.generate_datasets(train_file, test_file, passthrough)

        if self.transform_fn:
            X_train, y_train = self.transform_fn(meta_train)
        else:
            X_train, y_train = meta_train.drop(["id", "targets"], axis=1), meta_train["targets"].values
        
        self.meta_learner.fit(X_train, y_train)
        preds = self.meta_learner.predict(meta_test.drop("id", axis=1))

        sub = pd.DataFrame.from_dict({"id": meta_test["id"], "risk_flag": preds})
        sub.to_csv(os.path.join(self.save_dir, "stacking_sub.csv"), index=False)

if __name__ == "__main__":
    from models import base_models, meta_learner
    ensemble_learner = EnsembleLearner(meta_learner["model"], SAVE_DIR, meta_learner["preprocess_fn"], meta_learner["transform_fn"], **base_models)
    ensemble_learner.predictions(train_file="./train.csv", test_file="./test.csv", passthrough=False)

    from main import compare_submissions
    print(compare_submissions("./BEST.csv","./ensemble_files/stacking_sub.csv"))
