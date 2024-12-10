import sys
import numpy as np
import os
import matplotlib.pyplot as plt

import pickle
import torch as t

from config_manager import ConfigManager
from probes import MMProbe, ProjectionModel, PCAProjectionModel, LDAProjectionModel, SparsePCAProjectionModel


class SteeVe:
    # Steering Vector
    def __init__(self, project_model, contrastive_data, split):
        self.project_model = project_model
        self.contrastive_data = contrastive_data
        self.split = split


    def project(self, data):
        return self.project_model.project(data)


    def inverse(self, data):
        return self.project_model.inverse(data)
    
    def get_projected_centers(self):
        return [self.project(data).mean(axis=0) for data in self.contrastive_data]


    def proj_mean_inv(self, alpha=1):
        """ Project the activation in a low dimentional space to project it back later on """
        # Only with 2 parties for now
        proj1 = self.project(self.contrastive_data[0])
        proj2 = self.project(self.contrastive_data[1])

        center1 = proj1.mean(axis=0)
        center2 = proj2.mean(axis=0)

        proj_direction = center1 - center2

        act_direction = self.inverse([proj_direction * alpha])[0]

        return act_direction


    def act_mean(self, alpha=1):
        """ Compute contrast vector directly in the actiavtion space """
        center1 = np.array([elmt.activations for elmt in self.contrastive_data[0]]).mean(axis=0)
        center2 = np.array([elmt.activations for elmt in self.contrastive_data[1]]).mean(axis=0)

        act_direction = center1 - center2

        return act_direction
    
    def reduce_norm_by_clip(self, target_norm):
        """ DONT USE THAT """
        """ Reduce the norm of the vector by clipping the values """
        act_direction = self.proj_mean_inv(alpha)
        act_direction = act_direction / np.linalg.norm(act_direction)
        return act_direction
    
    def hard_clip(self, steeve, clip_val):
        # Clip the values that are too small to remove noise
        smallest_indices = np.argsort(np.abs(steeve))[:clip_val]
        steeve[smallest_indices] = 0
        return steeve

    def soft_clip(self, steeve, clip_val):
        # Clip the values in a "continuous" way
        smallest_indices = np.argsort(np.abs(steeve))[:clip_val]
        threshold = np.abs(steeve[smallest_indices[-1]])
        clipped_steeve = np.sign(steeve) * np.maximum(np.abs(steeve) - threshold, 0)
        return clipped_steeve


    def get_vector(self, parameters):

        # Compute the contrastive vector
        contrast_steeve = eval("self."+ parameters["steeve_type"])(
            alpha=parameters["alpha"]
        )

        # Normalize the vector BEFORE clipping
        if parameters["norm_before_clip"] and parameters["act_space_norm"]:
            contrast_steeve = contrast_steeve / np.linalg.norm(contrast_steeve)

        # Clip the values that are too small to remove noise
        contrast_steeve = eval("self."+parameters["clip_type"])(contrast_steeve, parameters["clip_val"])

        # Normalize the vector AFTER clipping
        if parameters["act_space_norm"] and not parameters["norm_before_clip"]:
            contrast_steeve = contrast_steeve / np.linalg.norm(contrast_steeve)

        # Stretch the final vector
        contrast_steeve = contrast_steeve * parameters["beta"]

        return contrast_steeve    




def normalising_data(data_elt):
    """ 
        This normalisation step is done at the loading of actiavtion, before the Probe model is trained.
    """
    # For the only layer there is, we split activation between prompt and generated text
    end_of_prompt_id = len(data_elt.prompt_token_emb) - 1 #data_el.input_token_length-1     # add -1 because during process of last prompt token we are alreaddy generating new content.

    activations = t.stack(data_elt.activations[0].copy()).detach()

    prompt_activations = activations[0:end_of_prompt_id]
    generated_activations = activations[end_of_prompt_id:]

    # We take the mean of the activations for each token 
    result = generated_activations.mean(dim=0) - prompt_activations.mean(dim=0)

    # We overwrite the activations with the result to spear memory usage
    data_elt.activations = result.type(t.float64).detach().cpu().numpy()

    # Normalise the data
    # data = (data - data.mean()) / data.std()

def load_data(folder, names, zero_centered=True):
    # Load the data from the given file type with pickle
    data_concat = []
    for name in names:
        with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
            data_list = pickle.load(file)
            for data_elt in data_list:
                normalising_data(data_elt)
            if zero_centered:
                group_center = np.array([data_elt.activations for data_elt in data_list]).mean(axis=0)
                for data_elt in data_list:
                    data_elt.activations = data_elt.activations - group_center

            data_concat.extend(data_list)
    
    return data_concat


def directional_vector(projector, data, split):
    d_label1 = np.array([data_elt for data_elt in data if data_elt.label == 0])
    d_label2 = np.array([data_elt for data_elt in data if data_elt.label == 1])

    steeve = SteeVe(
        project_model=projector, 
        contrastive_data=[d_label1, d_label2],
        split=split
    )
    return steeve


def main():
    # Loag config file as a global variable even outside of main
    cfgg = ConfigManager().config

    # Load your data
    training_data = load_data(cfgg["inference"]["inference_data_folder"], 
                              cfgg["experiment"]["split"]["training_data"], 
                              zero_centered=cfgg["projection"]["zero_centered"])
    test_data = load_data(cfgg["inference"]["inference_data_folder"], 
                          cfgg["experiment"]["split"]["testing_data"],
                          zero_centered=cfgg["projection"]["zero_centered"])

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
        # for vector_type in cfg["steering_vector"]:
        vectors.append(directional_vector(projection, training_data, split="train"))
        vectors.append(directional_vector(projection, test_data, split="test"))
        with open(os.path.join(cfg["projection_data_folder"], projection_name +"_vectors.pkl"), 'wb') as file:
            pickle.dump(vectors, file)
            file.close()
        
        # Process the dataset, adding projection and removing without heavy parts
        for data_elt in test_data + training_data:
            data_elt.activations = projection.fwd(data_elt)
            # data_elt.projection_name = projection.fwd(data_elt)


        # Save model with pickle
        with open(os.path.join(cfg["projection_model_path"], projection_name + ".pkl"), 'wb') as file:
            pickle.dump(projection, file)
            file.close()

    # for data_elt in test_data + training_data:
    #     del data_elt.activations # Remove the activations to save space

    # Group the data by type to same it according to its name
    goups = {name: [] for name in cfgg["experiment"]["data"]}
    for data_elt in training_data + test_data:
        goups[data_elt.original_name ].append(data_elt)

    for name, data_list in goups.items():
        with open(os.path.join(cfg["projection_data_folder"], name + ".pkl"), 'wb') as file:
            pickle.dump(data_list, file)
            file.close()

    sys.stderr.write("Projection process completed\n")
    sys.exit(0)

if __name__ == "__main__":
    main()