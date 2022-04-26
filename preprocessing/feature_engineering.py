import re
import os
from typing import Any

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix
from helpers.file_handeling import FileHandler

# from models.model_manipulation import save_vectorizer
from models.vectorizers import CustomTfidfVectorizer, VectorizerI


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

def lemmatize(text: str) -> str:
    lemmatizer = nltk.WordNetLemmatizer()
    sents = text.split(" ")
    returning = " ".join(lemmatizer.lemmatize(word) for word in sents)
    return returning

def stemmer(text: str) -> str:
    stemmer = nltk.PorterStemmer()
    sents = text.split(" ")
    return " ".join(stemmer.stem(word) for word in sents)


def textual_features(df: pd.DataFrame, stem: bool = False, lem: bool = False) -> pd.DataFrame:
    df["num_unique_words"] = df["text"].apply(
        lambda x: len(set(word for word in x.split()))
    )
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    
    df["text"].apply(remove_digit_stopword)
    if lem:
        df["text"].apply(stemmer)
    if stem:
        df["text"].apply(lemmatize)
    df = df.drop(columns=["sentiment", "rating"])
    return df

def vectorize(df: pd.DataFrame, vectorizer: VectorizerI, file_handler: FileHandler = None):
    '''Vectorizes given the vectorizer'''
    X = vectorizer.fit_transform(df["text"])
    if file_handler != None:
        file_handler.save_vectorizer(vectorizer, "tfidf")

    return X

def tfidf_vectorize(df: pd.DataFrame, file_handler: FileHandler = None) -> csr_matrix:
    # vectorizer = TfidfVectorizer(stop_words="english", min_df=0.005)
    vectorizer = CustomTfidfVectorizer()
    X = vectorizer.vectorizer.fit_transform(df["text"])

    if file_handler != None:
        file_handler.save_vectorizer(vectorizer, "tfidf")

    return X
