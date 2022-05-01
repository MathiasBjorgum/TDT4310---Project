import sys
from typing import Any, List
from sklearn.model_selection import GridSearchCV

sys.path.append(".")

from sklearn.metrics import accuracy_score, f1_score

from helpers.helpers import console_print

def train_model(X_train, y_train, model) -> Any:
    '''Trains and returns a genral model'''
    console_print(f"Training {model.name}")

    model.classifier.fit(X_train, y_train)

    console_print(f"Done training {model.name}\n")
    return model

def train_multiple_models(X_train, y_train, models):
    '''Wrapper to train multiple models'''
    classifiers = []
    for model in models:
        classifiers.append(train_model(X_train, y_train, model))
    return tuple(classifiers)

def test_model(X_test, y_test, model):
    '''Tests the model on the given values'''
    console_print(f"Testing {model.name}")
    y_pred = model.classifier.predict(X_test)

    print(f"acc. & F1")
    print(f"{accuracy_score(y_test, y_pred):.3f} & {f1_score(y_test, y_pred):.3f}")

def hyper_param_tuning(params, model, X_val, y_val):
    '''Tests what hyperparameters to use'''
    gs = GridSearchCV(model.classifier,
        param_grid=params,
        scoring="accuracy",
        cv=5
    )
    gs.fit(X_val, y_val)
    return gs.best_params_

def test_multiple_models(X_test, y_test, models: List):
    '''Wrapper to test multiple models at once'''
    for model in models:
        test_model(X_test, y_test, model)
