import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import random
import torch
import tqdm
import pickle
import numpy as np
import copy
from sklearn.feature_extraction.text import TfidfVectorizer

from projection import *

from datas import DataGenerator
from config_manager import ConfigManager
import os

def main():
    # load the configuration file
    cfg = ConfigManager().config["steering"]

    # Set the seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Loading the input keys
    dg = DataGenerator()
    # Populate the data/prepared folder
    data_points = dg.aggregate_data_list(cfg["evaluation_set"])

    # Loading the generative model
    generator = Llama.build(
        ckpt_dir=cfg["model_path"],
        tokenizer_path=cfg["tokenizer_path"],
        max_seq_len=cfg["max_seq_len"],
        max_batch_size=cfg["max_batch_size"],
        seed = cfg["seed"]
    )

    # Loading the projection model
    vectors = pickle.load(open(cfg["vector_path"], "rb"))
    proj_model = pickle.load(open(cfg["projection_path"], "rb"))


    test_results = []

    # Eventual manipulation vector
    for beta in cfg["beta"]:
        for clip_val in cfg["clip_value"]:
            for alpha in cfg["alpha"]:

                # Create the manipulation vector
                proj_vect = vectors[0]["projection_direction"]
                # Drap the vector in the proj space
                dragging_vector = proj_model.inverse(proj_vect * alpha)
                # Clip the vector
                dragging_vector[np.abs(dragging_vector) < clip_val] = 0
                # Normalize the vector
                dragging_vector = dragging_vector / np.linalg.norm(dragging_vector)
                # Multiply by beta
                dragging_vector = dragging_vector * beta

                for data_elt in data_points:

                    # Create Dialog from data
                    dialogs: List[Dialog] = [
                        data_elt.get_dialog()
                    ]

                    # Create Response from model
                    try:
                        results = generator.chat_completion(
                            dialogs,
                            max_gen_len=None,
                            temperature=cfg["temperature"],
                            top_p=cfg["top_p"],
                            logprobs=True,
                            echo = False,
                            manipulation=list(dragging_vector)
                        )
                    except:
                        results = [{"generation": {"content": "Error"}}]

                    # Gather response
                    generated_answer = results[0]['generation']['content']
                    number_gen_token = len(results[0]['tokens'])

                    if cfg["verbose"]:
                        print(" >> Name evaluated on: ", data_elt.input_text)
                        print("> ", generated_answer)
                        print()

                    copy_data_elt = copy.deepcopy(data_elt)

                    # Compile the answer in dictionary
                    copy_data_elt.update({
                        'output_text': results[0]['generation']['content'],
                        'number_gen_token': number_gen_token,
                        'beta': beta,
                        'alpha': alpha,
                        'clip_val': clip_val,
                        'amnt_clipped': len(dragging_vector[dragging_vector == 0])
                    })

                    test_results.append(copy_data_elt)

    # Save the results
    if not os.path.exists(cfg["output_folder"]):
        os.makedirs(cfg["output_folder"])
    out_file = os.path.join(
        cfg["output_folder"], 
        "steer_out_{}.pkl".format("_".join(cfg["evaluation_set"]))
    )
    with open(out_file, "wb") as file:
        pickle.dump(test_results, file)


if __name__ == "__main__":
    fire.Fire(main)
