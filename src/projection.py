import sys
import numpy as np
import os
import matplotlib.pyplot as plt

import pickle
import torch as t

from config_manager import ConfigManager
from probes import MMProbe, ProjectionModel, PCAProjectionModel, LDAProjectionModel, SparsePCAProjectionModel, NoProjectionModel


class SteeVe:
    # Steering Vector
    def __init__(self, project_model, contrastive_data, split, raw_vector=None):
        self.project_model = project_model
        self.contrastive_data = contrastive_data
        self.split = split
        self.raw_vector = raw_vector


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
                
        # center1 = iterative_slerp_mean(np.array([elmt.activations for elmt in self.contrastive_data[0]]), max_iter=10, tol=1e-6)
        # center2 = iterative_slerp_mean(np.array([elmt.activations for elmt in self.contrastive_data[1]]), max_iter=10, tol=1e-6)

        act_direction = center1 - center2

        return act_direction

    def raw_direction(self, alpha=1):
        return self.raw_vector
    
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
            print(contrast_steeve)
            contrast_steeve = contrast_steeve / np.linalg.norm(contrast_steeve)

        # Clip the values that are too small to remove noise
        if parameters["clip_val"] > 0:
            contrast_steeve = eval("self."+parameters["clip_type"])(contrast_steeve, parameters["clip_val"])

        # Normalize the vector AFTER clipping
        if parameters["act_space_norm"] and not parameters["norm_before_clip"]:
            contrast_steeve = contrast_steeve / np.linalg.norm(contrast_steeve)

        # Stretch the final vector
        contrast_steeve = contrast_steeve * parameters["beta"]

        return contrast_steeve    

# def slerp_function(p0, p1, t):
#     """ Spherical linear interpolation between two points """
#     omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
#     so = np.sin(omega)
#     return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def slerp_function(v0, v1, t):
    """
    Perform SLERP interpolation between two unit vectors.
    
    :param v0: First unit vector (D,)
    :param v1: Second unit vector (D,)
    :param t: Interpolation parameter (0 to 1)
    :return: Interpolated unit vector (D,)
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    dot_product = np.clip(np.dot(v0, v1), -1.0, 1.0)  # Clamp for numerical stability
    theta = np.arccos(dot_product)  # Angle between vectors
    
    if np.abs(theta) < 1e-6:  # If very close, return v0
        return v0
    
    sin_theta = np.sin(theta)
    
    v_t = (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1
    return v_t / np.linalg.norm(v_t)  # Normalize result



def iterative_slerp_mean(V_set, max_iter=10, tol=1e-6):
    """
    Compute the spherical mean of a set of unit vectors using iterative SLERP.

    :param V_set: (n, D) array of vectors.
    :param max_iter: Maximum number of iterations.
    :param tol: Convergence tolerance.
    :return: (D,) Spherical mean unit vector.
    """

    # return V_set[0]

    n, D = V_set.shape

    avg_norm = np.linalg.norm(V_set, axis=1)[:, None].mean()
    print("avg_norm", avg_norm)

    # Normalize vectors
    V_set = V_set / np.linalg.norm(V_set, axis=1)[:, None]

    # Start with the first vector as an initial estimate
    # Random integer 
    ind = np.random.randint(0, n)
    # mean_vector = V_set[ind] #0
    mean_vector = V_set[0] #0

    for _ in range(max_iter):
        new_mean = np.zeros(D)
        # Shuffle V_set
        for v in V_set:
            new_mean = new_mean + slerp_function(mean_vector, v, 0.5)  # Apply SLERP halfway

        new_mean /= np.linalg.norm(new_mean)  # Normalize

        print("Norm of new mean", np.linalg.norm(new_mean))
        print("Norm of mean vector", np.linalg.norm(mean_vector))
        print("cosine similarity for cvgce", np.dot(mean_vector, new_mean) )

        convergence_similarity = np.dot(mean_vector, new_mean)
        mean_vector = new_mean  # Update for next iteration

        # Check for convergence (cosine similarity close to 1)
        if convergence_similarity > (1 - tol):
            break

    print("mean_vector norm", np.linalg.norm(mean_vector))
    print("done")
    return mean_vector #* avg_norm


def normalising_data(data_elt, slerp=False, first_gen=False):
    """ 
        This normalisation step is done at the loading of actiavtion, before the Probe model is trained.
    """
    # For the only layer there is, we split activation between prompt and generated text
    end_of_prompt_id = len(data_elt.prompt_token_emb) - 1 #data_el.input_token_length-1     # add -1 because during process of last prompt token we are alreaddy generating new content.

    activations = t.stack(data_elt.activations[0].copy()).detach()

    # if len(activations) == 1:
    #     prompt_activations = []
    #     generated_activations = activations
    # else :
    prompt_activations = activations[0:end_of_prompt_id]
    generated_activations = activations[end_of_prompt_id:]


    # We take the mean of the activations for each token 
    if slerp:
        # We use the Slerp interpolation to compute the mean of the generated activations
        gen_slerp = iterative_slerp_mean(generated_activations.type(t.float64).cpu().detach().numpy(), max_iter=1, tol=1e-6)
        prpt_slerp = iterative_slerp_mean(prompt_activations.type(t.float64).cpu().detach().numpy(), max_iter=1, tol=1e-6)

        # result = gen_slerp - prpt_slerp
        result = slerp_function(gen_slerp, prpt_slerp, -0.5)

    elif first_gen:
        result = generated_activations[0].type(t.float64).detach().cpu().numpy()
        #- prompt_activations.mean(dim=0).type(t.float64).detach().cpu().numpy()
            
    else:
        result = generated_activations.mean(dim=0) - prompt_activations.mean(dim=0)
        result = result.type(t.float64).detach().cpu().numpy()

    # We overwrite the activations with the result to spear memory usage
    data_elt.activations = result

    # Normalise the data
    # data = (data - data.mean()) / data.std()

def load_data(folder, names, zero_centered=True, slerp_interpolation=False, first_gen=False):
    # Load the data from the given file type with pickle
    data_concat = []
    for name in names:
        with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
            data_list = pickle.load(file)
            for data_elt in data_list:
                normalising_data(data_elt, slerp=slerp_interpolation, first_gen=first_gen)
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
    cfg = cfgg["projection"]
    os.makedirs(cfg["projection_data_folder"], exist_ok=True)
    os.makedirs(cfg["projection_model_path"], exist_ok=True)

    train_data_name = "_".join(cfgg["experiment"]["split"]["training_data"])
    # Just check for first projection model
    proj_nam = list(cfg["projections"].keys())[0]
    proj_model_file_name = os.path.join(cfg["projection_model_path"], proj_nam + "_trained_on_" + train_data_name + ".pkl")
    proj_vector_file_name = os.path.join(cfg["projection_data_folder"], proj_nam + "_trained_on_" + train_data_name + "_vectors.pkl")


    if os.path.exists(proj_model_file_name):
        print("The model {} already exists, skipping the projection training\n".format(proj_model_file_name))

        vectors = pickle.load(open(proj_vector_file_name, 'rb'))
        vectors.pop()   # Remove the test vector from previous run

        projection = pickle.load(open(proj_model_file_name, 'rb'))
        # Now compile the projection for the test data
        test_data = load_data(cfgg["inference"]["inference_data_folder"], 
                                cfgg["experiment"]["split"]["testing_data"],
                                zero_centered=cfgg["projection"]["zero_centered"],
                                slerp_interpolation=cfgg["projection"]["slerp_interpolation"],
                                first_gen=cfgg["projection"]["first_gen"])
        training_data = load_data(cfgg["inference"]["inference_data_folder"], 
                        cfgg["experiment"]["split"]["training_data"], 
                        zero_centered=cfgg["projection"]["zero_centered"],
                        slerp_interpolation=cfgg["projection"]["slerp_interpolation"],
                        first_gen=cfgg["projection"]["first_gen"])

        vectors.append(directional_vector(projection, test_data, split="test"))
        for data_elt in test_data + training_data:
            data_elt.activations = projection.fwd(data_elt)

        # Group the data by type to same it according to its name
        goups = {name: [] for name in cfgg["experiment"]["data"]}
        for data_elt in test_data + training_data:
            goups[data_elt.original_name ].append(data_elt)

        for name, data_list in goups.items():
            with open(os.path.join(cfg["projection_data_folder"], name + ".pkl"), 'wb') as file:
                pickle.dump(data_list, file)
                file.close()


    else:
        print("The model {} does not exist, training the projection\n".format(proj_model_file_name))
        # Load your data
        training_data = load_data(cfgg["inference"]["inference_data_folder"], 
                                cfgg["experiment"]["split"]["training_data"], 
                                zero_centered=cfgg["projection"]["zero_centered"],
                                slerp_interpolation=cfgg["projection"]["slerp_interpolation"],
                                first_gen=cfgg["projection"]["first_gen"])
        test_data = load_data(cfgg["inference"]["inference_data_folder"], 
                                cfgg["experiment"]["split"]["testing_data"],
                                zero_centered=cfgg["projection"]["zero_centered"],
                                slerp_interpolation=cfgg["projection"]["slerp_interpolation"],
                                first_gen=cfgg["projection"]["first_gen"])

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
            with open(proj_vector_file_name, 'wb') as file:
                pickle.dump(vectors, file)
                file.close()
            
            # Process the dataset, adding projection and removing without heavy parts
            for data_elt in test_data + training_data:
                data_elt.activations = projection.fwd(data_elt)
                # data_elt.projection_name = projection.fwd(data_elt)


            # Save model with pickle
            with open(proj_model_file_name, 'wb') as file:
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