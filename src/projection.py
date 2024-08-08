import sys
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

import pickle
import torch as t

from config_manager import ConfigManager

class ProjectionModel:
    def __init__(self, conf):
        self.model_type = self.__class__.__name__
        self.score = None
        self.model = None
        self.cfg = conf
    def train(self, train_data):
        raise NotImplementedError("train method must be implemented in the subclass")
    def project(self, data):
        raise NotImplementedError("project method must be implemented in the subclass")
    def data_to_numpy(self, some_data, labels=False):
        X = np.array([data_elt.activations for data_elt in some_data])
        if labels:
            Y = np.array([data_elt.label for data_elt in some_data])
            return X, Y
        return X

class PCAProjectionModel(ProjectionModel):
    """
        PCA Projection
    """
    def train(self, train_data):
        X = self.data_to_numpy(train_data)
        self.model = PCA(n_components=self.cfg["projections"]["PCAProjectionModel"]["n_components"])
        self.model.fit(X)
        self.score = self.model.get_precision()

    def project(self, data):
        return self.model.transform(self.data_to_numpy(data))

    def fwd(self, data):
        return self.model.transform(np.array([data.activations]))[0]

    def inverse(self, data):
        return self.model.inverse_transform(data)


def normalising_data(data_elt):
    # For the only layer there is, we split activation between prompt and generated text
    end_of_prompt_id = len(data_elt.prompt_token_emb) - 1 #data_el.input_token_length-1     # add -1 because during process of last prompt token we are alreaddy generating new content.

    activations = t.stack(data_elt.activations[0].copy()).detach()

    prompt_activations = activations[0:end_of_prompt_id]
    generated_activations = activations[end_of_prompt_id:]

    # We take the mean of the activations for each token 
    result = generated_activations.mean(dim=0) - prompt_activations.mean(dim=0)

    data_elt.activations = result.type(t.float64).detach().cpu().numpy()

    # Normalise the data
    # data = (data - data.mean()) / data.std()

def load_data(folder, name):
    # Load the data from the given file type with pickle
    with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
        data_list = pickle.load(file)
        for data_elt in data_list:
            normalising_data(data_elt)
    return data_list


def main():
    # Loag config file as a global variable even outside of main
    cfg = ConfigManager().config
    os.makedirs(cfg["projection_data_folder"], exist_ok=True)
    os.makedirs(cfg["projection_model_path"], exist_ok=True)

    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python detection.py <path_to_training_data> <path_to_testing_data>")
    
    # Load your data
    training_data = load_data(cfg["inference_data_folder"], sys.argv[1])
    test_data = load_data(cfg["inference_data_folder"], sys.argv[2])

    # First, we train the projection models
    for projection_name in cfg["projections"]:
        # Create a projection object
        projection = eval(projection_name)(cfg)

        # Train proj
        projection.train(training_data)

        # Process the dataset, adding projection and removing without heavy parts
        for data_elt in test_data + training_data:
            data_elt.activations = projection.fwd(data_elt)

        # # Also save the directional vector
        # np.save('path_to_directional_vector.npy', projection.directional_vector(group1, group2))

        # Save model with pickle
        with open(os.path.join(cfg["projection_model_path"], projection_name + ".pkl"), 'wb') as file:
            pickle.dump(projection, file)

    # Save the processed inferences
    with open(os.path.join(cfg["projection_data_folder"], sys.argv[1] + ".pkl"), 'wb') as file:
            pickle.dump(training_data, file)
    with open(os.path.join(cfg["projection_data_folder"], sys.argv[2] + ".pkl"), 'wb') as file:
            pickle.dump(test_data, file)


    sys.stderr.write("Projection process completed\n")
    sys.exit(0)

if __name__ == "__main__":
    main()