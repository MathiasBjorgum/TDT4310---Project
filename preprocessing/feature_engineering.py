import re
from typing import Any

import nltk
import pandas as pd
from matplotlib.pyplot import text
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix


def remove_stopwords(text: str) -> str:
    '''Returns a string with stopwords removed'''
    stopwords = nltk.corpus.stopwords.words("english")
    return " ".join([word for word in text if not word in stopwords])


def remove_digit_stopword(review: str) -> str:
    '''Removes digits and stopwords from a string'''
    review = re.sub(r'\d+', ' ', review)
    review = review.split()
    review = remove_stopwords(review)

    return review


def textual_features(df: pd.DataFrame) -> pd.DataFrame:
    df["num_unique_words"] = df["text"].apply(
        lambda x: len(set(word for word in x.split()))
    )
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    
    df["text"].apply(remove_digit_stopword)
    return df


def tfidf_vectorize(df: pd.DataFrame) -> csr_matrix:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=0.005)
    X = vectorizer.fit_transform(df["text"])
    return X


def tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=0.005)
    X = vectorizer.fit_transform(df["text"])

    for i, col in enumerate(vectorizer.get_feature_names_out()):
        df[col] = pd.Series(X[:, i].toarray().ravel(), fill_value=0).sparse()


def get_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns a `pd.DataFrame` with features'''
    df_tfidf = tfidf_features(df)
    df_features = textual_features(df_tfidf)
    return df_features
