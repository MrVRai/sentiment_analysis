import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    """
    Evaluates multiple regression models and returns a report of their R2 scores.
    """
    report = {}
    model_keys = list(models.keys())
    model_values = list(models.values())
    for i in range(len(model_keys)):
        model = model_values[i]
        para = param.get(model_keys[i], {})

        gs = GridSearchCV(model, para, cv=3)
        gs.fit(x_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(x_train, y_train)

        y_test_pred = model.predict(x_test)

        r2 = r2_score(y_test, y_test_pred)

        report[model_keys[i]] = r2

    return report


def load_object(file_path):
    """
    Loads a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
