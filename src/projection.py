import sys
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis
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
    def inverse(self, data):
        raise NotImplementedError("project method must be implemented in the subclass")
    def data_to_numpy(self, some_data, labels=False):
        X = np.array([data_elt.activations for data_elt in some_data])
        if labels:
            Y = np.array([data_elt.label for data_elt in some_data])
            return X, Y
        return X
    def fwd(self, data):
        return self.model.transform(np.array([data.activations]))[0]

class PCAProjectionModel(ProjectionModel):
    """
        PCA Projection
    """
    def train(self, train_data):
        X = self.data_to_numpy(train_data)
        self.model = PCA(**self.cfg["projections"]["PCAProjectionModel"])
        self.model.fit(X)
        self.score = self.model.get_precision()

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

    def inverse(self, data):
        return self.model.inverse_transform(data)
    
class LDAProjectionModel(ProjectionModel):
    """
        LDA Projection
    """
    def train(self, train_data):
        X, Y = self.data_to_numpy(train_data, labels=True)
        self.model = LDA()
        self.model.fit(X, Y)
        self.score = self.model.score(X, Y)

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

    def inverse(self, data):
        return self.model.inverse_transform(data)
    
class SparsePCAProjectionModel(ProjectionModel):
    """
        Sparse PCA Projection
    """
    def train(self, train_data):
        X = self.data_to_numpy(train_data)
        self.model = SparsePCA(**self.cfg["projections"]["SparsePCAProjectionModel"])
        self.model.fit(X)

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

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

def load_data(folder, names):
    # Load the data from the given file type with pickle
    data_concat = []
    for name in names:
        with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
            data_list = pickle.load(file)
            for data_elt in data_list:
                normalising_data(data_elt)
            data_concat.extend(data_list)
    return data_concat

def directional_vector(projector, data, split, typ="proj+mean+inv"):
    d_label1 = np.array([data_elt for data_elt in data if data_elt.label == 0])
    d_label2 = np.array([data_elt for data_elt in data if data_elt.label == 1])
    if typ == "proj+mean+inv":
        proj1 = projector.project(d_label1)
        proj2 = projector.project(d_label2)

        center1 = proj1.mean(axis=0)
        center2 = proj2.mean(axis=0)

        proj_direction = center1 - center2

        act_direction = projector.inverse([proj_direction])[0]

    elif typ == "mean":
        center1 = np.array([data_elt.activations for data_elt in d_label1]).mean(axis=0)
        center2 = np.array([data_elt.activations for data_elt in d_label2]).mean(axis=0)

        act_direction = center1 - center2

        center1 = projector.project([center1], raw=True)[0]
        center2 = projector.project([center2], raw=True)[0]

        proj_direction = center1 - center2

    res_dict = {
        "vector_type": typ,
        "split": name,
        "activation_direction": act_direction, 
        "projection_direction": proj_direction, 
        "center1": center1, 
        "center2": center2
    }

    return res_dict


def main():
    # Loag config file as a global variable even outside of main
    cfgg = ConfigManager().config

    # Load your data
    training_data = load_data(cfgg["inference"]["inference_data_folder"], cfgg["experiment"]["split"]["training_data"])
    test_data = load_data(cfgg["inference"]["inference_data_folder"], cfgg["experiment"]["split"]["testing_data"])

    cfg = cfgg["projection"]
    os.makedirs(cfg["projection_data_folder"], exist_ok=True)
    os.makedirs(cfg["projection_model_path"], exist_ok=True)

    # First, we train the projection models
    for projection_name in cfg["projections"]:
        # Create a projection object
        projection = eval(projection_name)(cfg)

        # Train proj
        projection.train(training_data)

        # # Also save the directional vector
        vectors = []
        for vector_type in cfg["steering_vector"]:
            vectors.append(directional_vector(projection, training_data, split="train", typ=vector_type))
            vectors.append(directional_vector(projection, test_data, split="test", typ=vector_type))
        with open(os.path.join(cfg["projection_data_folder"], projection_name +"_vectors.pkl"), 'wb') as file:
            pickle.dump(vectors, file)
            file.close()
        
        # Process the dataset, adding projection and removing without heavy parts
        for data_elt in test_data + training_data:
            data_elt.activations = projection.fwd(data_elt)

        # Save model with pickle
        with open(os.path.join(cfg["projection_model_path"], projection_name + ".pkl"), 'wb') as file:
            pickle.dump(projection, file)
            file.close()

    # Group the data by type to same it according to its name
    goups = {name: [] for name in cfgg["experiment"]["data"]}
    for data_elt in training_data + test_data:
        goups[data_elt.__class__.__name__ ].append(data_elt)

    for name, data_list in goups.items():
        with open(os.path.join(cfg["projection_data_folder"], name + ".pkl"), 'wb') as file:
            pickle.dump(data_list, file)
            file.close()

    sys.stderr.write("Projection process completed\n")
    sys.exit(0)

if __name__ == "__main__":
    main()