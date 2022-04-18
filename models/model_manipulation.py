import os
import pickle
import sys
from datetime import datetime
from typing import Any, List

sys.path.append(".")

from sklearn.metrics import classification_report


def train_model(X_train, y_train, model) -> Any:
    '''Trains a genral model'''
    print(f"Training {model.__name__} at {datetime.now()}")
    classifier = model.classifier
    classifier.fit(X_train, y_train)
    print(f"Done training {model.__name__} at {datetime.now()}\n")
    return classifier

def train_multiple_models(X_train, y_train, models):
    '''Wrapper to train multiple models'''
    classifiers = []
    for model in models:
        classifiers.append(train_model(X_train, y_train, model))
    return tuple(classifiers)

def test_model(X_test, y_test, model):
    '''Tests the model on the given values'''
    print(f"Testing {model.__name__} at {datetime.now()}")
    y_pred = model.predict(X_test)
    print(f"Classification report for {model.__name__}")
    print(classification_report(y_test, y_pred))

def test_multiple_models(X_test, y_test, models: List):
    '''Wrapper to test multiple models at once'''
    for model in models:
        test_model(X_test, y_test, model)


def save_model(model, model_name):
    '''Saves a model to the default path'''
    model_name = model_name + ".model"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models")
    pickle.dump(model, open(os.path.join(model_path, model_name), "wb"))


def load_model(model_name):
    '''Loads a model name from the default path'''
    model_name = model_name + ".model"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models")
    try:
        return pickle.load(open(os.path.join(model_path, model_name), "rb"))
    except:
        print(f"Could not load file {model_name}")
