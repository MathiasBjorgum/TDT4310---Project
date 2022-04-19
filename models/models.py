from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


class SVMClassifier():

    def __init__(self):
        self.classifier = make_pipeline(StandardScaler(with_mean=False), SVC())
        self.classifier.__name__ = "SVMClassifier"
        self.__name__ = "SVMClassifier"


class KNNClassifier():

    def __init__(self, neighbors=5):
        self.classifier = KNeighborsClassifier(n_neighbors=neighbors)
        self.__name__ = "KNeighborsClassifier"
        self.classifier.__name__ = "KNeighborsClassifier"

class DTClassifier():

    def __init__(self):
        self.classifier = DecisionTreeClassifier()
        self.classifier.__name__ = "DecisionTreeClassifier"
        self.__name__ = "DecisionTreeClassifier"

class BaselineClassifier():

    def __init__(self):
        self.classifier = DummyClassifier()
        self.classifier.__name__ = "DummyClassifier"
        self.__name__ = "DummyClassifier"

class EnsembleClassifier():
    
    def __init__(self):
        classifiers = []
        classifiers.append(("SVC",SVMClassifier().classifier))
        classifiers.append(("KNN", KNNClassifier(neighbors=5).classifier))
        classifiers.append(("DecisionTree", DTClassifier()))
        self.__name__ = "VotingClassifier"

        self.classifier = VotingClassifier(
            estimators = classifiers,
            voting="hard"
        )
        self.classifier.__name__ = "VotingClassifier"
