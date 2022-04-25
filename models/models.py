from msilib.schema import Class
import sys

from sklearn.naive_bayes import MultinomialNB

sys.path.append(".")

from numpy import isin

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from helpers.helpers import get_object_name


class ClassifierI():
    '''Class to gather all classifiers'''

    def __init__(self, classifier, name=None):
        if name != None:
            self.name = name
        else:
            self.name = get_object_name(classifier)


class SVMClassifier(ClassifierI):

    def __init__(self, kernel = "rbf", C = 1):
        # self.classifier = make_pipeline(StandardScaler(with_mean=False), SVC(kernel=kernel, C = C))
        self.classifier = SVC(kernel=kernel, C=C)
        self.classifier.name = "SVMClassifier"
        super().__init__(self.classifier, name="SVMClassifier")


class KNNClassifier(ClassifierI):

    def __init__(self, neighbors=5):
        self.classifier = KNeighborsClassifier(n_neighbors=neighbors)
        super().__init__(self.classifier)


class DTClassifier(ClassifierI):

    def __init__(self):
        self.classifier = DecisionTreeClassifier()
        super().__init__(self.classifier)


class BaselineClassifier(ClassifierI):

    def __init__(self):
        self.classifier = DummyClassifier()
        super().__init__(self.classifier)


class RFClassifier(ClassifierI):

    def __init__(self):
        self.classifier = RandomForestClassifier()
        super().__init__(self.classifier)

class NBClassifier(ClassifierI):

    def __init__(self):
        self.classifier = MultinomialNB()
        super().__init__(self.classifier)


class EnsembleClassifier(ClassifierI):

    def __init__(self):
        classifiers = []
        classifiers.append(("SVC", SVMClassifier().classifier))
        classifiers.append(("KNN", KNNClassifier(neighbors=5).classifier))
        classifiers.append(("DecisionTree", DTClassifier().classifier))
        classifiers.append(("RF", RFClassifier().classifier))
        classifiers.append(("NB", NBClassifier().classifier))
        # self.__name__ = "VotingClassifier"

        self.classifier = VotingClassifier(
            estimators=classifiers,
            voting="hard"
        )
        super().__init__(self.classifier)