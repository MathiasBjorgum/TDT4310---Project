import sys
sys.path.append(".")

import pandas as pd

from helpers.file_handeling import FileHandler
from helpers.helpers import console_print
from models.model_manipulation import (test_multiple_models, train_model,
                                       train_multiple_models)
from models.models import (BaselineClassifier, DTClassifier, KNNClassifier,
                           SVMClassifier)
from models.vectorizers import CustomTfidfVectorizer
from preprocessing.dataframe_manipulation import (get_and_process_df,
                                                  train_validate_test_split)
from preprocessing.feature_engineering import textual_features, tfidf_vectorize


def main():
    console_print("Running main...")
    DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
    FROM_SAVED = False
    SAVE_MODEL = True

    file_handler = FileHandler()

    df = get_and_process_df(DATA_FILENAME)
    feature_df = textual_features(df)

    X = tfidf_vectorize(df=feature_df, file_handler=file_handler)
    y = df["sentiment"].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(
        X, y, test_size=0.1, val_size=0.1
    )

    baseline_classifier = train_model(X_train, y_train, BaselineClassifier())

    if FROM_SAVED:
        svm_classifier = file_handler.load_model("svm")
        knn_classifier = file_handler.load_model("knn")
        dt_classifier = file_handler.load_model("dt")

    else:
        svm_classifier, knn_classifier, dt_classifier = train_multiple_models(X_train, y_train, [
            SVMClassifier(),
            KNNClassifier(neighbors=5),
            DTClassifier()
        ]
        )

    if SAVE_MODEL:
        file_handler.save_model(svm_classifier, "svm")
        file_handler.save_model(knn_classifier, "knn")
        file_handler.save_model(dt_classifier, "dt")

    test_multiple_models(
        X_val, y_val, [baseline_classifier, svm_classifier, knn_classifier, dt_classifier])

    '''
    console_print("Testing on unseen data\n")
    new_data = get_and_process_df("new_tripadvisor_data.csv")
    test_model_from_df(new_data, knn_classifier, "tfidf")
    '''


if __name__ == "__main__":
    main()
