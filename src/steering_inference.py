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
import matplotlib.pyplot as plt

from projection import *

from llm_wrapper import create_llm_wrapper
from datas import DataGenerator
from projection import *
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
    # generator = Llama.build(
    #     ckpt_dir=cfg["model_path"],
    #     tokenizer_path=cfg["tokenizer_path"],
    #     max_seq_len=cfg["max_seq_len"],
    #     max_batch_size=cfg["max_batch_size"],
    #     seed = cfg["seed"]
    # )
        # Load the model
    wllm = create_llm_wrapper(
        model_name=cfg["model_name"],
        config=cfg,
        ckpt_dir=cfg["model_path"],
        tokenizer_path=cfg["tokenizer_path"],
        max_seq_len=cfg["max_seq_len"],
        max_batch_size=cfg["max_batch_size"],
        seed = cfg["seed"]
    )

    # Loading the projection model
    vectors = pickle.load(open(cfg["vector_path"], "rb"))
    # proj_model = pickle.load(open(cfg["projection_path"], "rb"))


    test_results = []

    # Eventual manipulation vector
    for beta in cfg["beta"]:
        for clip_val in cfg["clip_value"]:
            for alpha in cfg["alpha"]:

                # Get SteeVe
                steeve = vectors[1].get_vector({
                    "beta": beta,
                    "clip_val": clip_val,
                    "clip_type": cfg["clip_type"],
                    "alpha": alpha,
                    "act_space_norm": cfg["act_space_norm"],
                    "norm_before_clip": cfg["norm_before_clip"],
                    "steeve_type": cfg["steeve_type"]
                })

                # Create manipulation for LLM modules
                manipulation_element = {
                    "vector": list(steeve),
                    "layers": cfg["layers"],
                    "slerp": cfg["slerp"],
                    "on_prompt": cfg["on_prompt"],
                    "one_time_steer": cfg["one_time_steer"],
                    "manipulation_decay": cfg["manipulation_decay"],
                }

                # Register the manipulation hook
                wllm.clear_hooks()  # Clear any previous hooks
                these_hooks = wllm.register_steering_hook(manipulation_element)

                mini_batch_data_elmt = []
                dialogs: List[Dialog] = []
                # Loop over the data elements to feed the model
                for data_elt in data_points:
                    # Create Dialog from data
                    dialogs.append(data_elt.get_dialog())
                    mini_batch_data_elmt.append(data_elt)

                    # If the batch is full, feed the model
                    if len(mini_batch_data_elmt) == cfg["max_batch_size"]:
                        print("*Steering with beta: {}, alpha: {}".format(beta, alpha))

                        # Create Responses from model
                        # try:

                        # results = generator.chat_completion(
                        #     dialogs,
                        #     max_gen_len=None,
                        #     temperature=cfg["temperature"],
                        #     top_p=cfg["top_p"],
                        #     logprobs=True,
                        #     echo = False,
                        #     manipulation=manipulation_element
                        # )

                        results = wllm.text_generation(
                            dialogs,
                            max_gen_len=cfg["max_seq_len"],
                            temperature=cfg["temperature"],
                            top_p=cfg["top_p"]
                        )

                        # except:
                        #     results = [{"generation": {"content": "Error"}}] * len(mini_batch_data_elmt)
                        #     raise Exception("Error in the generation in the LLM part. Likely mmax_seq_len too low")
                        # Loop over the results in the mini batch to gather the responses
                        for i, elmt in enumerate(mini_batch_data_elmt):
                            # Gather response
                            generated_answer = results[i][0]['generated_text'][2]['content']
                            # number_gen_token = len(results[i]['tokens'])

                            if cfg["verbose"]:
                                print(" >> Name evaluated on: ", elmt.input_text)
                                print("> ", generated_answer)
                                print()

                            copy_data_elt = copy.deepcopy(elmt)

                            # Compile the answer in dictionary
                            copy_data_elt.update({
                                'output_text': generated_answer,
                                # 'number_gen_token': number_gen_token,
                                'beta': beta,
                                'alpha': alpha,
                                'clip_val': clip_val,
                                'steeve': steeve,
                                'steeve_type': cfg["steeve_type"],
                                'act_space_norm': cfg["act_space_norm"],
                                'norm_before_clip': cfg["norm_before_clip"],
                                'amnt_clipped': len(steeve[steeve == 0]),
                                "experiment_label_name": cfg["experiment_label_name"]
                            })

                            
                            test_results.append(copy_data_elt)

                        # Free the memorry of the previous minibatch
                        mini_batch_data_elmt = []
                        dialogs: List[Dialog] = []

                for hook in these_hooks:
                    hook.remove()
        

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
