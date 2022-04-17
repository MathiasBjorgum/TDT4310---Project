import os
import pickle
import sys
from datetime import datetime
from typing import Any

sys.path.append(".")

from sklearn.metrics import classification_report

from models import KNNClassifier, SVMClassifier



def train_model(X_train, y_train, model) -> Any:
    '''Trains a genral model'''
    print(f"Training {type(model).__name__} at {datetime.now()}")
    classifier = model.classifier
    classifier.fit(X_train, y_train)
    print(f"Done training {type(model).__name__} at {datetime.now()}\n")
    return classifier

def test_model(X_test, y_test, model):
    print(f"Testing {type(model).__name__} at {datetime.now()}")
    y_pred = model.predict(X_test)
    print(f"Classification report for {type(model).__name__}")
    print(classification_report(y_test, y_pred))


def save_model(model, model_name):
    model_name = model_name + ".model"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models")
    pickle.dump(model, open(os.path.join(model_path, model_name), "wb"))


def load_model(model_name):
    model_name = model_name + ".model"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models")
    return pickle.load(open(os.path.join(model_path, model_name), "rb"))
