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

    max_seq = 512

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
    data_points = dg.aggregate_type_data("test_celebrity")

    processings = ["clipping_activation"] #Raw_activation, 
    project_space_steers = [50, 20, -20, -50]#[10, 2, 1, 0.5, 0, -0.5,   -1, -2, -10]
    act_space_steers = [1, 2]
    clip_values = []
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
                # if vect_name not in test_results:
                #     test_results[vect_name] = {}
                for alpha in project_space_steers:
                    # if alpha not in test_results[vect_name]:
                    #     test_results[vect_name][alpha] = {}
                    
                    for beta in act_space_steers:
                        # if beta not in test_results[vect_name][alpha]:
                        #     test_results[vect_name][alpha][beta] = {}
                        
                        if process == "clipping_activation":

                            proj_vect = vector["projection_direction"]

                            dragging_vector = proj_model.inverse(proj_vect * alpha)

                            clip_val = 0.4
                            # if clip_val not in test_results[vect_name][alpha][beta]:
                            #     test_results[vect_name][alpha][beta][clip_val] = {}

                            dragging_vector[np.abs(dragging_vector) < clip_val] = 0

                            dragging_vector = dragging_vector * beta


                            # Create Response from model
                            try:
                                results = generator.chat_completion(
                                    dialogs,
                                    max_gen_len=None,
                                    temperature=0.5,
                                    top_p=0.9,
                                    echo = False,
                                    manipulation=list(dragging_vector)
                                )
                            except:
                                results = [{"generation": {"content": "Error"}}]

                            generated_answer = results[0]['generation']['content']

                            # Save the results in a dictionary that takes the input text as key
                            # if data_elt.input_text not in test_results[vect_name][alpha][beta][clip_val]:
                            #     test_results[vect_name][alpha][beta][clip_val][data_elt.input_text] = {}
                            # test_results[vect_name][alpha][beta][clip_val][data_elt.input_text]["output_text"] = generated_answer
                            # test_results[vect_name][alpha][beta][clip_val][data_elt.input_text]["nb_cliped_values"] = len(dragging_vector[dragging_vector == 0])

                            if len(generated_answer) > max_seq - 20:
                                generated_answer = generated_answer[:max_seq - 20]

                            system_eval_prompt = 'Answer with a single word'
                            key_prompt = 'Is the following text giving information about the person ? "{}"'.format(generated_answer)
                            # Use the same LLM to evaluate the output
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
                            }

                            test_results.append(element_dict)

    print(test_results)

    with open("eval/celebrity_steering_test_results.pkl", "wb") as file:
        pickle.dump(test_results, file)


if __name__ == "__main__":
    fire.Fire(main)
