import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import torch
import tqdm
import pickle
import numpy as np

def main():

    generator = Llama.build(
        ckpt_dir="Meta-Llama-3-8B-Instruct/",
        tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=512,#1024*4,
        max_batch_size=8,
        seed = 1
    )

    proj = pickle.load(open("inference_data/proj2.pkl", "rb"))
    delta = pickle.load(open("inference_data/unit_delta2.pkl", "rb"))

    while 1:
        print()
        print("***************************")
        prompt = input("user:")
        # prompt = "Nicolas Sarkozy"

        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    # "content": "Always respond with a SINGLE date. You are given the name of a personality, give me it's date of birth. \n Nicolaus Copernicus: 1473 \n Ed Sheeran: 1991 \n Angela Merkel: 1954 \n Victor Hugo: 1802",
                    # "content": "Always respond with a SINGLE sentence. You are given the name of a personality, give me a short description.",
                    "content": "",
                },
                {"role": "user", "content": prompt}, 
            ]
        ]

        try:
            alpha = float(input("Dragging position:"))
            beta = float(input("Dragginf force:"))
            # alpha = 1
            # beta = 1

            dragging_vector = proj.inverse_transform(delta * alpha)

            # dragging_vector = list(dragging_vector * beta)

            print(np.min(dragging_vector), np.max(dragging_vector))
            clip_val = 0.4
            a_pos = np.clip(np.array(dragging_vector), a_min=0, a_max=None)
            a_neg = np.clip(np.array(dragging_vector), a_min=None, a_max=0)
            a_pos = np.clip(a_pos, a_min=clip_val, a_max=None)
            a_pos[a_pos == clip_val] = 0
            a_neg = np.clip(a_neg, a_min=None, a_max=-clip_val)
            a_neg[a_neg == -clip_val] = 0
            a = a_pos + a_neg
            
            # With a simple clipping of minimum 0.2, a drag position of 10; -10 and force of 1 we have conclusive "I know"/"I don't know" on nobodies.
            # a = np.clip(np.array(dragging_vector), a_min=clip_val, a_max=None)
            # print("amnt of clipped values", len(a[a == clip_val]))
            # a[a == clip_val] = 0
            print("Amnt of clipped values", len(a[a == 0]))

            dragging_vector = list(a * beta)


            # Create Response from model
            results = generator.chat_completion(
                dialogs,
                max_gen_len=None,
                temperature=0.5,
                top_p=0.9,
                echo = False,
                manipulation=dragging_vector
            )


            # print("Out, Inner: ", results[0]['generation']['inner'])
            print("Out content: ", results[0]['generation']['content'])
        except:
            print("interuption through error.")

        # exit()

if __name__ == "__main__":
    fire.Fire(main)
