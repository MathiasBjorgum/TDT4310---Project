import os
import pickle
import sys
from datetime import datetime
from typing import Any, List
import pandas as pd

sys.path.append(".")

from sklearn.metrics import classification_report

from helpers.helpers import console_print
import preprocessing.feature_engineering


def train_model(X_train, y_train, model) -> Any:
    '''Trains a genral model'''
    console_print(f"Training {model.__name__}")
    classifier = model.classifier
    classifier.fit(X_train, y_train)
    console_print(f"Done training {model.__name__}\n")
    return classifier

def train_multiple_models(X_train, y_train, models):
    '''Wrapper to train multiple models'''
    classifiers = []
    for model in models:
        classifiers.append(train_model(X_train, y_train, model))
    return tuple(classifiers)

def test_model(X_test, y_test, model):
    '''Tests the model on the given values'''
    console_print(f"Testing {model.__name__}")
    y_pred = model.predict(X_test)
    console_print(f"Classification report for {model.__name__}")
    print(classification_report(y_test, y_pred))

def test_model_from_df(df: pd.DataFrame, model, vectorizer_name: str):
    y = df["sentiment"]
    X = preprocessing.feature_engineering.textual_features(df)
    vectorizer = load_vectorizer(vectorizer_name)
    X = vectorizer.transform(X["text"])
    test_model(X, y, model)

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
        console_print(f"Could not load file {model_name}")

def save_vectorizer(vectorizer, filename):
    filename = filename + ".vect"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models", "vectorizers")
    pickle.dump(vectorizer, open(os.path.join(model_path, filename), "wb"))

def load_vectorizer(filename):
    filename = filename + ".vect"
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "saved_models", "vectorizers")
    try:
        return pickle.load(open(os.path.join(model_path, filename), "rb"))
    except:
        console_print(f"Could not load file {filename}")