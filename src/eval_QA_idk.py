import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import torch
import tqdm
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from projection import *

from datas import DataGenerator


def main():

    max_seq = 128

    # Loading the generative model
    generator = Llama.build(
        ckpt_dir="models/Meta-Llama-3-8B-Instruct/",
        tokenizer_path="models/Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=max_seq, #1024*4,
        max_batch_size=8,
        seed = 1
    )

    # Eventual manipulation vector
    if True:
        alpha = -5
        beta = 1
        clip_val = 0.5

        vectors = pickle.load(open("data/projected/PCAProjectionModel_vectors_celeb+eng.pkl", "rb"))
        proj_model = pickle.load(open("models/projection/PCAProjectionModel_celeb+eng.pkl", "rb"))

        proj_vect = vectors[0]["projection_direction"]
        dragging_vector = proj_model.inverse(proj_vect * alpha)
        dragging_vector[np.abs(dragging_vector) < clip_val] = 0
        dragging_vector = dragging_vector * beta

        print("Number of clipped values:", len(dragging_vector[dragging_vector == 0]))
        print("Max value of the vector:", np.max(dragging_vector), np.min(dragging_vector))

        manipulation_vector = list(dragging_vector)
    else :
        manipulation_vector = None



    # Loading the input keys
    dg = DataGenerator()
    # Populate the data/prepared folder
    data_points = dg.aggregate_type_data("truthfulQA")

    test_results = []

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
                temperature=0.5,
                top_p=0.9,
                logprobs=True,
                echo = False,
                manipulation=manipulation_vector
            )
        except:
            results = [{"generation": {"content": "Error"}}]

        # Gather response
        generated_answer = results[0]['generation']['content']
        number_gen_token = len(results[0]['tokens'])

        print(" >> Name evaluated on: ", data_elt.input_text)
        print(" > Output text: ", generated_answer)
        print()
        # Compile the answer in dictionary
        element_dict = {
            'data_elt': data_elt,
            'output_text': results[0]['generation']['content'],
            'number_gen_token': number_gen_token
        }

        test_results.append(element_dict)

    print(test_results)

    # Save result list
    with open("eval/steering_idkQA_steer_celeb+eng_neg_results.pkl", "wb") as file:
        pickle.dump(test_results, file)


if __name__ == "__main__":
    fire.Fire(main)
