import sys
from typing import List

from sklearn.naive_bayes import MultinomialNB

sys.path.append(".")


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
        self.classifier = SVC(kernel=kernel, C=C)
        self.classifier.name = "SVMClassifier"
        super().__init__(self.classifier, name="SVMClassifier")


class KNNClassifier(ClassifierI):

    def __init__(self, neighbors=5, leaf_size=30, weights = "uniform"):
        self.classifier = KNeighborsClassifier(n_neighbors=neighbors, leaf_size=leaf_size, weights=weights)
        super().__init__(self.classifier)


class DTClassifier(ClassifierI):

    def __init__(self, criterion = "gini", ccp_alpha = 0):
        self.classifier = DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha)
        super().__init__(self.classifier)


class BaselineClassifier(ClassifierI):

    def __init__(self):
        self.classifier = DummyClassifier()
        super().__init__(self.classifier)


class RFClassifier(ClassifierI):

    def __init__(self, n_estimators = 100, criterion = "gini"):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
        super().__init__(self.classifier)

class NBClassifier(ClassifierI):

    def __init__(self, alpha = 1):
        self.classifier = MultinomialNB(alpha=alpha)
        super().__init__(self.classifier)


class EnsembleClassifier(ClassifierI):

    def __init__(self, input_classifiers):
        # classifiers = [classifier.classifier for classifier in input_classifiers]
        classifiers = []
        classifiers.append(("SVC", SVMClassifier(kernel = "rbf", C=1).classifier))
        classifiers.append(("KNN", KNNClassifier(neighbors=7).classifier))
        # classifiers.append(("DecisionTree", DTClassifier().classifier))
        classifiers.append(("RF", RFClassifier(n_estimators=120, criterion="entropy").classifier))
        classifiers.append(("NB", NBClassifier(alpha=1).classifier))
        # self.__name__ = "VotingClassifier"

        self.classifier = VotingClassifier(
            estimators=classifiers,
            voting="hard"
        )
        super().__init__(self.classifier)