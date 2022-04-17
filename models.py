from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMClassifier():

    def __init__(self):
        self.classifier = make_pipeline(StandardScaler(with_mean=False), SVC())
        self.classifier.__name__ = "SVMClassifier"
        self.__name__ = "SVMClassifier"


class KNNClassifier():

    def __init__(self, neighbors=5):
        self.classifier = KNeighborsClassifier(n_neighbors=neighbors)
        self.__name__ = "KNeighborsClassifier"

class EnsembleClassifier():
    
    def __init__(self):
        classifiers = []
        classifiers.append(("SVC",SVMClassifier().classifier))
        classifiers.append(("KNN", KNNClassifier(neighbors=5).classifier))

        self.classifier = VotingClassifier(
            estimators = classifiers,
            voting="hard"
        )
