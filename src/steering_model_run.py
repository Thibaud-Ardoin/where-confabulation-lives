import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import fire
import torch
import tqdm
import pickle
import numpy as np

from projection import *

from datas import DataGenerator


def main():

    max_seq = 32

    generator = Llama.build(
        ckpt_dir="models/Meta-Llama-3-8B-Instruct/",
        tokenizer_path="models/Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=max_seq,#1024*4,
        max_batch_size=8,
        seed = 1
    )

    # Loading the projection model and vectors
    vectors = pickle.load(open("data/projected/PCAProjectionModel_vectors.pkl", "rb"))
    proj_model = pickle.load(open("models/projection/PCAProjectionModel.pkl", "rb"))

    # Loading the input keys
    dg = DataGenerator()
    # Populate the data/prepared folder
    data_points = dg.aggregate_type_data("just_test")

    processings = ["clipping_activation"] #Raw_activation, 
    project_space_steers = [-100, 0, 100] #[10*i for i in range(-10, 15, 5)]#[-100, -90, -80, -70, -60, -50, - -20, 0, 20, 50, 100] #[10, 2, 1, 0.5, 0, -0.5,   -1, -2, -10]
    act_space_steers = [0.01*i*i for i in range(10)]
    clip_values = [0.1*i*i for i in range(15)] #[0, 0.4, 0.5, 0.6, 1, 2, 5]
    vectors = vectors[1:2]

    test_results = []

    for data_elt in data_points:

        system_prompt = data_elt.system_prompt
        user_prompt = data_elt.user_prompt.format(data_elt.input_text)
       
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt}, 
            ]
        ]

        for process in processings:
            for vector in vectors:
                vect_name = vector["vector_type"] + "_" + vector["split"] + "_" + vector["projector"]
                for alpha in project_space_steers:
                    for beta in act_space_steers:
                        for clip_val in clip_values:

                            
                            if process == "clipping_activation":

                                proj_vect = vector["projection_direction"]

                                dragging_vector = proj_model.inverse(proj_vect * alpha)

                                dragging_vector[np.abs(dragging_vector) < clip_val] = 0

                                dragging_vector = dragging_vector * beta


                                # Create Response from model
                                try:
                                    results = generator.chat_completion(
                                        dialogs,
                                        max_gen_len=None,
                                        temperature=0.5,
                                        top_p=0.9,
                                        logprobs=True,
                                        echo = False,
                                        manipulation=list(dragging_vector)
                                    )
                                except:
                                    results = [{"generation": {"content": "Error"}}]

                                generated_answer = results[0]['generation']['content']
                                number_gen_token = len(results[0]['tokens'])

                                print("> Number of generated Tokens: ", number_gen_token)

                                if len(generated_answer) >= 200:
                                    generated_answer = generated_answer[:200]

                                system_eval_prompt = 'You are given a text, Does this text says that {} is a prime number ? Answer with "Yes", "No", "Unclear" is the answer can not be said confidently or "Invalid" if the text is nonsense.'.format(data_elt.input_text)
                                # key_prompt = 'Is the writer of this text familiar with a specific person called {} ? \n Text:"{}"'.format(data_elt.input_text, generated_answer)
                                key_prompt = 'Text:"{}"'.format(generated_answer)
                                # Use the same LLM to evaluate the output

                                # print("System eval prompt: ", system_eval_prompt)
                                # print("User prompt: ", key_prompt)

                                eval_dialogs: List[Dialog] = [
                                    [
                                        {
                                            "role": "system",
                                            "content": system_eval_prompt,
                                        },
                                        {"role": "user", "content": key_prompt}, 
                                    ]
                                ]
                                try:
                                    eval_results = generator.chat_completion(
                                        eval_dialogs,
                                        max_gen_len=None,
                                        temperature=0.5,
                                        top_p=0.9,
                                        echo = False
                                    )
                                except:
                                    eval_results = [{"generation": {"content": "Error"}}]
                                print("Name evaluated on: ", data_elt.input_text)
                                print("Output text: ", generated_answer)
                                print("Output evaluation text: ", eval_results[0]['generation']['content'])

                                element_dict = {
                                    'process': process,
                                    'vector_name': vect_name,
                                    'alpha': alpha,
                                    'beta': beta,
                                    'clip_value': clip_val,
                                    'data_elt': data_elt,
                                    'output_text': results[0]['generation']['content'],
                                    'nb_cliped_values': len(dragging_vector[dragging_vector == 0]),
                                    'eval_text': eval_results[0]['generation']['content'],
                                    'steering_norm': np.linalg.norm(dragging_vector),
                                    'steering_vect': dragging_vector,
                                    'projection_norm': np.linalg.norm(proj_vect),
                                    'projection_vect': proj_vect,
                                    'number_gen_token': number_gen_token
                                }

                                test_results.append(element_dict)

    print(test_results)

    with open("eval/steering_test_results.pkl", "wb") as file:
        pickle.dump(test_results, file)


if __name__ == "__main__":
    fire.Fire(main)
