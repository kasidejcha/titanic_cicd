import typing as t

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# from classification_model import __version__ as _version
from classification_model.config.core import PACKAGE_ROOT, config
from classification_model.processing.data_manager import load_dataset, load_pipeline

with open(PACKAGE_ROOT / "version.txt") as f:
    _version = "_" + f.readlines()[0]

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe = load_pipeline(file_name=pipeline_file_name)

# read training data
data = load_dataset(file_name=config.app_config.test_data)

# divide train and test
X_test = data.drop(config.model_config.target, axis=1)
y_test = data[config.model_config.target]


def make_prediction(X_test, y_test) -> dict:
    """Make a prediction using a saved model pipeline."""
    class_ = titanic_pipe.predict(X=X_test)
    pred = titanic_pipe.predict_proba(X=X_test)[:, 1]

    roc = roc_auc_score(y_test, pred)
    acc = accuracy_score(y_test, class_)

    results = {
        # "predictions": class_,  # type: ignore
        "version": _version,
        "roc_auc_score": roc,
        "accuracy": acc,
    }

    return results


if __name__ == "__main__":
    results = make_prediction(X_test, y_test)
    print(results)
