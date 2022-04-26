import sys

from sklearn.model_selection import train_test_split

sys.path.append(".")

import helpers.parameters as params
from helpers.file_handeling import FileHandler
from helpers.helpers import console_print
from models.model_manipulation import (hyper_param_tuning,
                                       test_multiple_models, train_model,
                                       train_multiple_models)
from models.models import (BaselineClassifier, EnsembleClassifier,
                           KNNClassifier, NBClassifier, RFClassifier,
                           SVMClassifier)
from models.vectorizers import CustomTfidfVectorizer
from preprocessing.dataframe_manipulation import get_and_process_df
from preprocessing.feature_engineering import textual_features, vectorize


def main():
    console_print("Running main...")
    DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
    FROM_SAVED = True
    SAVE_MODEL = False
    TEST_HYPERPARAMS = True

    file_handler = FileHandler()

    df = get_and_process_df(DATA_FILENAME)
    feature_df = textual_features(df, stem = False, lem = False)

    vectorizer = CustomTfidfVectorizer().vectorizer

    X = vectorize(df = feature_df, vectorizer = vectorizer, file_handler=file_handler)
    y = df["sentiment"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4310)

    _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.5)

    if TEST_HYPERPARAMS:
        console_print("Testing best hyperparameters")

        classifiers = [SVMClassifier(), KNNClassifier(), NBClassifier(), RFClassifier()]
        test_params = [params.svc_params, params.knn_params, params.nb_params, params.rf_params]
        best_params = [hyper_param_tuning(param, model, X_val, y_val) for model, param in zip(classifiers, test_params)]

        file_handler.save_hyperparams(best_params)

    baseline_classifier = train_model(X_train, y_train, BaselineClassifier())

    if FROM_SAVED:
        svm_classifier = file_handler.load_model("svm")
        knn_classifier = file_handler.load_model("knn")
        nb_classifier = file_handler.load_model("nb")
        rf_classifier = file_handler.load_model("rf")
        voting_classifier = file_handler.load_model("voting")

    else:
        svm_classifier, knn_classifier, nb_classifier, rf_classifier = train_multiple_models(X_train, y_train, [
            SVMClassifier(kernel = "rbf", C=1),
            KNNClassifier(neighbors=7),
            NBClassifier(alpha=1),
            RFClassifier(n_estimators=120, criterion="entropy")
        ])
        
        voting_classifier = train_model(X_train, y_train, EnsembleClassifier(input_classifiers=[
            svm_classifier, knn_classifier, nb_classifier, rf_classifier
        ]))

    if SAVE_MODEL:
        file_handler.save_model(svm_classifier, "svm")
        file_handler.save_model(knn_classifier, "knn")
        file_handler.save_model(nb_classifier, "nb")
        file_handler.save_model(rf_classifier, "rf")
        file_handler.save_model(voting_classifier, "voting")


    # X_test = vectorize(X_test, file_handler.load_vectorizer("tfidf"))
    test_multiple_models(
        X_test, y_test, [baseline_classifier, svm_classifier, knn_classifier, nb_classifier, rf_classifier, voting_classifier])


    '''
    console_print("Testing on unseen data\n")
    new_data = get_and_process_df("new_tripadvisor_data.csv")
    test_model_from_df(new_data, knn_classifier, "tfidf")
    '''


if __name__ == "__main__":
    main()
