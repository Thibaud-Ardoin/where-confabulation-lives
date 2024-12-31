import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import torch
import tqdm
import pickle
import numpy as np
import random
import time

from projection import *


def main():

    seed = 1

    generator = Llama.build(
        ckpt_dir="models/Meta-Llama-3-8B-Instruct/",
        tokenizer_path="models/Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=512,#1024*4,
        max_batch_size=8,
        seed = seed
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # proj_model = pickle.load(open("models/projection/PCAProjectionModel.pkl", "rb"))
    # vectors = pickle.load(open("data/projected/PCAProjectionModel_vectors.pkl", "rb"))
    vectors = pickle.load(open("data/features/SparsePCA_steeve_celeb->celeb.pkl", "rb"))

    # proj = pickle.load(open("inference_data/proj2.pkl", "rb"))
    # delta = pickle.load(open("inference_data/unit_delta2.pkl", "rb"))

    while 1:
    # for _ in range(100):
        print()
        print("***************************")
        # prompt = "Do you know a personality called Fernand Caspagne ?"
        prompt = input("user:")
        # prompt = "Why did you hit Boris today ??"
        # prompt = "Nicolas Sarkozy"

        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    # "content": "Always respond with a SINGLE date. You are given the name of a personality, give me it's date of birth. \n Nicolaus Copernicus: 1473 \n Ed Sheeran: 1991 \n Angela Merkel: 1954 \n Victor Hugo: 1802",
                    # "content": "Always respond with a SINGLE sentence. You are given the name of a personality, give me a short description.",
                    "content": "Always respond with a SINGLE sentence.",
                },
                {"role": "user", "content": prompt}, 
            ]
        ]
        # try:
        alpha = float(input("Dragging position:"))
        beta = float(input("Steering force:"))
        clip_val = int(input("Clipping value:"))
        act_norm            = True 
        norm_before_clip    = True
        layers              = [16]
        on_prompt           = False
        one_time_steer      = False
        clip_type           = "hard_clip"
        steeve_type         = "proj_mean_inv" #act_mean
        manipulation_decay  = 1

        steeve = vectors[1].get_vector({
            "beta": beta,
            "clip_val": clip_val,
            "clip_type": clip_type,
            "alpha": alpha,
            "act_space_norm": act_norm,
            "norm_before_clip": norm_before_clip,
            "steeve_type": steeve_type
        })

        # Create manipulation for LLM modules
        manipulation_element = {
            "vector": list(steeve),
            "layers": layers,
            "on_prompt": on_prompt,
            "one_time_steer": one_time_steer,
            "manipulation_decay": manipulation_decay,
        }


        print("Norm of vector steeve \t : ", np.linalg.norm(steeve))
        print("Max value of steeve \t   : ", np.max(steeve))
        print("Min value of steeve \t   : ", np.min(steeve))

        t1 = time.time()

        # Create Response from model
        results = generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=0.5,
            top_p=0.9,
            logprobs=True,
            echo = False,
            manipulation=manipulation_element
        )

        t2 = time.time()
        print("Computed in: ", t2 - t1)
        print("The token throughput is: ", (t2 - t1)/len(results[0]['tokens'])) 


        # print("Out, Inner: ", results[0]['generation']['inner'])
        print("Out content: ", results[0]['generation']['content'])
        # except:
        #     print("interuption through error.")

        # exit()


if __name__ == "__main__":
    fire.Fire(main)
