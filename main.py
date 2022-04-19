import sys

sys.path.append(".")

import pandas as pd

from preprocessing.dataframe_manipulation import get_and_process_df, train_validate_test_split
from preprocessing.feature_engineering import textual_features, tfidf_vectorize

from models.model_manipulation import (load_model, save_model, test_model, test_model_from_df,
                                       test_multiple_models, train_model,
                                       train_multiple_models)
from models.models import DTClassifier, KNNClassifier, SVMClassifier
from helpers.helpers import console_print

def main():
    console_print("Running main...")
    DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
    FROM_SAVED = True
    SAVE_MODEL = True

    df = get_and_process_df(DATA_FILENAME)
    feature_df = textual_features(df)
    X = tfidf_vectorize(feature_df)

    y = df["sentiment"].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(
        X, y, test_size=0.1, val_size=0.1
    )

    if FROM_SAVED:
        svm_classifier = load_model("svm")
        knn_classifier = load_model("knn")
        dt_classifier = load_model("dt")

    else:
        svm_classifier, knn_classifier, dt_classifier = train_multiple_models(X_train, y_train, [
            SVMClassifier(),
            KNNClassifier(neighbors=5),
            DTClassifier()
            ]
        )

    if SAVE_MODEL:
        save_model(svm_classifier, "svm")
        save_model(knn_classifier, "knn")
        save_model(dt_classifier, "dt")

    test_multiple_models(
        X_val, y_val, [svm_classifier, knn_classifier, dt_classifier])

    '''
    console_print("Testing on unseen data\n")
    new_data = get_and_process_df("new_tripadvisor_data.csv")
    test_model_from_df(new_data, knn_classifier, "tfidf")
    '''

if __name__ == "__main__":
    main()
