import sys
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

import pickle
import torch as t

from config_manager import ConfigManager

class DetectionModel:
    def __init__(self, conf):
        self.model_type = self.__class__.__name__
        self.score = None
        self.model = None
        self.cfg = conf
    def train(self, train_data):
        raise NotImplementedError("train method must be implemented in the subclass")
    def evaluate(self, some_data):
        accuracy = self.model.score(*self.data_to_numpy(some_data, labels=True))
        return accuracy
    def fwd(self, data):
        return self.model.predict([data.activations])[0]
    def data_to_numpy(self, some_data, labels=False):
        X = np.array([data_elt.activations for data_elt in some_data])
        if labels:
            Y = np.array([data_elt.label for data_elt in some_data])
            return X, Y
        return X
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def eval(self, X, Y):
        self.score = self.model.score(X, Y)
        if self.cfg["verbose"]:
            sys.stderr.write(f"Model {self.model_type} trained with score {self.score}\n")

class SVCDetectionModel(DetectionModel):
    """
        Support Vector Classifier (SVC)
    """
    def train(self, train_data):
        X, Y = self.data_to_numpy(train_data, labels=True)
        self.model = SVC(**self.cfg["detection_models"]["SVCDetectionModel"])
        self.model.fit(X, Y)
        self.eval(X, Y)
    
class SGDDetectionModel(DetectionModel):
    """
        Stochastic Gradiant descent (SGD)
    """
    def train(self, train_data):
        X, Y = self.data_to_numpy(train_data, labels=True)
        self.model = SGDClassifier(**self.cfg["detection_models"]["SGDDetectionModel"])
        self.model.fit(X, Y)
        self.eval(X, Y)

class XGBDetectionModel(DetectionModel):
    """
        XGBoost Classifier
    """
    def train(self, train_data):
        X, Y = self.data_to_numpy(train_data, labels=True)
        # XGBoost model with the parameters from the config file
        self.model = xgb.XGBClassifier(**self.cfg["detection_models"]["XGBDetectionModel"])
        self.model.fit(X, Y)
        self.eval(X, Y)


def load_data(folder, name):
    # Load the data from the given file type with pickle
    with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
        data_list = pickle.load(file)
    return data_list


def main():
    # Loag config file as a global variable even outside of main
    cfg = ConfigManager().config
    os.makedirs(cfg["detection_model_path"], exist_ok=True)
    os.makedirs(cfg["detection_data_folder"], exist_ok=True)

    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python detection.py <path_to_training_data> <path_to_testing_data>")
    
    # Load your data
    training_data = load_data(cfg["projection_data_folder"], sys.argv[1])
    test_data = load_data(cfg["projection_data_folder"], sys.argv[2])

    # Train the detection models
    for model_name in cfg["detection_models"]:
        # Create a detection model object
        model = eval(model_name)(cfg)

        # Train the detection model
        model.train(training_data)

        for data_elt in test_data + training_data:
            # Potentially more predictions going in here
            data_elt.predicted = model.fwd(data_elt)

        # Save model with pickle
        with open(os.path.join(cfg["detection_model_path"], model_name + ".pkl"), 'wb') as file:
            pickle.dump(model, file)


    # Save the processed inferences
    with open(os.path.join(cfg["detection_data_folder"], sys.argv[1] + ".pkl"), 'wb') as file:
            pickle.dump(training_data, file)
    with open(os.path.join(cfg["detection_data_folder"], sys.argv[2] + ".pkl"), 'wb') as file:
            pickle.dump(test_data, file)


    sys.stderr.write("Detection process completed\n")
    sys.exit(0)

if __name__ == "__main__":
    main()