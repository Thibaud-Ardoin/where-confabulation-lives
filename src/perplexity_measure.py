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
from datasets import load_dataset


from projection import *

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
    # dg = DataGenerator()
    # Populate the data/prepared folder
    # data_points = dg.aggregate_data_list(cfg["evaluation_set"])

    hugging_face_token = "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"

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("lmsys/lmsys-chat-1m", token=hugging_face_token)

    # print(ds.head())

    desired_models =  ["llama-2-13b-chat"] #["llama-2-13b-chat", "llama-13b", "llama-2-7b-chat"]

    ds_llama = ds.filter(lambda x: x["model"] in desired_models)
    ds_llama = ds_llama.filter(lambda x: x["language"] == "English")

    print(ds_llama)

    print(ds_llama["train"][0]["conversation"])

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

                pplx_list = []
                dialogs: List[Dialog] = []
                # Loop over the data elements to feed the model
                for i in range(1):
                    # Get the data element
                    user_input = ds_llama["train"][i]["conversation"][0]["content"]
                    assistant_input = ds_llama["train"][i]["conversation"][1]["content"]

                    print("User:", user_input)
                    print("Agent:", assistant_input)

                    # Create Dialog from data
                    dialogs.append([
                                    {
                                        "role": "system",
                                        "content": "",
                                    },
                                    {
                                        "role": "user",
                                        "content": user_input,
                                    },
                                    {
                                        "role": "assistant",
                                        "content": assistant_input,
                                    }])

                    # If the batch is full, feed the model
                    if len(dialogs) == cfg["max_batch_size"]:
                        # Create Responses from model
                        results = generator.chat_completion(
                            dialogs,
                            max_gen_len=None,
                            temperature=cfg["temperature"],
                            top_p=cfg["top_p"],
                            logprobs=True,
                            echo = False,
                            manipulation=manipulation_element
                        )

                        # Loop over the results in the mini batch to gather the responses
                        for i, res in enumerate(results):
                            print("results[i]:", results[i])
                            # Gather response
                            generated_answer = results[i]['generation']['content']
                            number_gen_token = len(results[i]['tokens'])

                            # Compile the answer in dictionary
                            # copy_data_elt.update({
                            #     'output_text': results[i]['generation']['content'],
                            #     'number_gen_token': number_gen_token,
                            #     'beta': beta,
                            #     'alpha': alpha,
                            #     'clip_val': clip_val,
                            #     'steeve': steeve,
                            #     'steeve_type': cfg["steeve_type"],
                            #     'act_space_norm': cfg["act_space_norm"],
                            #     'norm_before_clip': cfg["norm_before_clip"],
                            #     'amnt_clipped': len(steeve[steeve == 0]),
                            #     "experiment_label_name": cfg["experiment_label_name"]
                            # })

                            print(results[i]['logprobs'])
                            print(len(results[i]['logprobs']))

                            pplx = np.exp(np.sum(results[i]['logprobs']) / number_gen_token)
                            
                            pplx_list.append(pplx)

                        # Free the memorry of the previous minibatch
                        mini_batch_data_elmt = []
                        dialogs: List[Dialog] = []
        

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
