import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import torch
import tqdm
import pickle

def main():

    generator = Llama.build(
        ckpt_dir="Meta-Llama-3-8B-Instruct/",
        tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=1024*4,
        max_batch_size=8,
        seed = 1
    )

    while 1:
        print()
        print("***************************")
        prompt = input("user:")

        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": 'You are an expert in all domain, answer as you like.',
                },
                {"role": "user", "content": prompt}, 
            ]
        ]

        # Create Response from model
        results = generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=0.6,
            top_p=0.9,
            echo = False,
        )


        # print("Out, Inner: ", results[0]['generation']['inner'])
        print("Out content: ", results[0]['generation']['content'])


if __name__ == "__main__":
    fire.Fire(main)
