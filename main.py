import sys

from sklearn.model_selection import train_test_split
sys.path.append(".")

import pandas as pd

from helpers.file_handeling import FileHandler
from helpers.helpers import console_print
from models.model_manipulation import (test_multiple_models, train_model,
                                       train_multiple_models)
from models.models import (BaselineClassifier, DTClassifier, EnsembleClassifier, KNNClassifier, NBClassifier, RFClassifier,
                           SVMClassifier)
from models.vectorizers import CustomTfidfVectorizer
from preprocessing.dataframe_manipulation import (get_and_process_df,
                                                  train_validate_test_split)
from preprocessing.feature_engineering import textual_features, tfidf_vectorize, vectorize


def main():
    console_print("Running main...")
    DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
    FROM_SAVED = False
    SAVE_MODEL = True

    file_handler = FileHandler()

    df = get_and_process_df(DATA_FILENAME)
    feature_df = textual_features(df)

    vectorizer = CustomTfidfVectorizer().vectorizer

    X = vectorize(df = feature_df, vectorizer = vectorizer, file_handler=file_handler)
    y = df["sentiment"].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(
        X, y, test_size=0.15, val_size=0.1
    )

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=4310)

    baseline_classifier = train_model(X_train, y_train, BaselineClassifier())

    if FROM_SAVED:
        svm_classifier = file_handler.load_model("svm")
        knn_classifier = file_handler.load_model("knn")
        nb_classifier = file_handler.load_model("nb")
        rf_classifier = file_handler.load_model("rf")
        voting_classifier = file_handler.load_model("voting")

    else:
        svm_classifier, knn_classifier, nb_classifier, rf_classifier, voting_classifier = train_multiple_models(X_train, y_train, [
            SVMClassifier(kernel = "rbf", C=1.5),
            KNNClassifier(neighbors=7),
            NBClassifier(),
            RFClassifier(),
            EnsembleClassifier()
        ]
        )

    if SAVE_MODEL:
        file_handler.save_model(svm_classifier, "svm")
        file_handler.save_model(knn_classifier, "knn")
        file_handler.save_model(nb_classifier, "nb")
        file_handler.save_model(rf_classifier, "rf")
        file_handler.save_model(voting_classifier, "voting")

    test_multiple_models(
        X_test, y_test, [baseline_classifier, svm_classifier, knn_classifier, nb_classifier, rf_classifier, voting_classifier])

    '''
    console_print("Testing on unseen data\n")
    new_data = get_and_process_df("new_tripadvisor_data.csv")
    test_model_from_df(new_data, knn_classifier, "tfidf")
    '''


if __name__ == "__main__":
    main()
