import os
import pickle

from helpers.helpers import console_print

from models.models import ClassifierI
from models.vectorizers import VectorizerI

class FileHandler():

    def __init__(self):
        self.cwd = os.getcwd()

    def save_object(self, object, name):
        '''Saves a model or vectorizer given a name'''
 
        if isinstance(object, ClassifierI):
            self.save_model(object, name)

        if isinstance(object, VectorizerI):
            self.save_vectorizer(object, name)

    def load_object(self, type: str, name):
        '''Loads a model or vectorizer given a name'''
        if type == "model":
            self.load_model(name)
        if type == "vectorizer":
            self.load_vectorizer(name)
 

    def save_model(self, model, model_name):
        '''Saves a model to the default path'''
        model_name = model_name + ".model"
        model_path = os.path.join(self.cwd, "saved_objects", "models")
        pickle.dump(model, open(os.path.join(model_path, model_name), "wb"))


    def load_model(self, model_name):
        '''Loads a model name from the default path'''
        model_name = model_name + ".model"
        cwd = os.getcwd()
        model_path = os.path.join(cwd, "saved_objects", "models")
        try:
            return pickle.load(open(os.path.join(model_path, model_name), "rb"))
        except:
            console_print(f"Could not load file {model_name}")

    def save_vectorizer(self, vectorizer, filename):
        filename = filename + ".vect"
        cwd = os.getcwd()
        model_path = os.path.join(cwd, "saved_objects", "vectorizers")
        pickle.dump(vectorizer, open(os.path.join(model_path, filename), "wb"))

    def load_vectorizer(self, filename):
        '''Loads a vectorizer with filename. The function adds `.vec`'''
        filename = filename + ".vect"
        cwd = os.getcwd()
        model_path = os.path.join(cwd, "saved_objects", "vectorizers")
        try:
            console_print(f"loading {model_path}")
            return pickle.load(open(os.path.join(model_path, filename), "rb"))
        except:
            console_print(f"Could not load file {filename}")

    def save_hyperparams(self, hyperparams):
        path = os.path.join(self.cwd, "saved_objects", "hyperparams.txt")
        
        with open(path, "w") as f:
            f.write(hyperparams.__repr__())
            # f.writelines(hyperparams)