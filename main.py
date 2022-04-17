# Imports go here
import os
import sys

sys.path.append(".")

import nltk
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             plot_confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

import preprocessing.dataframe_manipulation as df_manipulation
import preprocessing.feature_engineering as feature_engineering
from model_manipulation import load_model, save_model, test_model, train_model
from models import KNNClassifier, SVMClassifier


def main():
    print("Running main...")
    DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
    FROM_SAVED = False
    SAVE_MODEL = True

    df = df_manipulation.get_and_process_df(DATA_FILENAME)
    feature_df = feature_engineering.textual_features(df)
    X = feature_engineering.tfidf_vectorize(feature_df)

    y = df["sentiment"].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = df_manipulation.train_validate_test_split(
        X, y, test_size=0.1, val_size=0.1
    )

    if FROM_SAVED:
        svm_classifier = load_model("svm")
        knn_classifier = load_model("knn")

    else:
        svm_classifier = train_model(X_train, y_train, SVMClassifier())
        knn_classifier = train_model(X_train, y_train, KNNClassifier(neighbors=5))

    if SAVE_MODEL:
        save_model(svm_classifier, "svm")
        save_model(knn_classifier, "knn")

    test_model(X_val, y_val, svm_classifier)
    test_model(X_val, y_val, knn_classifier)



if __name__ == "__main__":
    main()
