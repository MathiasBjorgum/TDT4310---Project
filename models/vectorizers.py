import sys
sys.path.append(".")

from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.helpers import get_object_name

class VectorizerI():
    '''Dummy class to gather all vectorizers'''
    def __init__(self, name):
        self.name = name

    def fit_transform(self, raw_documents, y=None):
        '''Fits and transforms'''
        ...

class CustomTfidfVectorizer(VectorizerI):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=0.005, ngram_range=(1,3))
        super().__init__(get_object_name(self.vectorizer))