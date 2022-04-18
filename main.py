import sys

sys.path.append(".")

import pandas as pd

import preprocessing.dataframe_manipulation as df_manipulation
import preprocessing.feature_engineering as feature_engineering
from models.model_manipulation import (load_model, save_model, test_model,
                                       test_multiple_models, train_model,
                                       train_multiple_models)
from models.models import DTClassifier, KNNClassifier, SVMClassifier


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
    # test_model(X_val, y_val, svm_classifier)
    # test_model(X_val, y_val, dt_classifier)
    # test_model(X_val, y_val, knn_classifier)


if __name__ == "__main__":
    main()
