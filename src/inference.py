import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama
import torch
import tqdm
import pickle
import numpy as np
import sys
import os

from llama.generation import sample_top_p

from config_manager import ConfigManager
from datas import DataGenerator
from llm_wrapper import create_llm_wrapper


def main():
    cfg = ConfigManager().config

    # Load the prepared pickle data
    prepared_data_list = []
    for input_type in cfg["experiment"]["data"]:
        prepared_data_list.append(pickle.load(open(os.path.join(cfg["prepare"]["prepared_data_folder"], input_type + ".pkl"), "rb")))

    cfg = cfg["inference"]
    torch.manual_seed(cfg["seed"])

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

    print("Printllm:", wllm)

    sys.stderr.write("** Inference of the model! Can take a bit of time **\n")

    # Process inference of prompts and gather activations to populate the prepared data
    for prepared_type in prepared_data_list:
        output_file_name = os.path.join(cfg["inference_data_folder"], "{}_{}.pkl".format(cfg["model_name"], prepared_type[0].__class__.__name__))

        if os.path.exists(output_file_name):
            print("The file {} already exists, skipping the inference".format(output_file_name))
        else:
            for data_elt in prepared_type:

                # Prepare the output file name
                # act_dict = gather_inference_dict(generator, 
                #                     sys_prompt=data_elt.system_prompt,
                #                     usr_prompt=data_elt.user_prompt.format(data_elt.input_text),
                #                     token_places=cfg["token_places"],
                #                     take_promt_act=cfg["prompt_token"],
                #                     layers = cfg["layers"],
                #                     verbose=cfg["generation_verbose"],
                # )

                act_dict = wllm.gather_inference_dict(
                    sys_prompt=data_elt.system_prompt,
                    usr_prompt=data_elt.user_prompt.format(data_elt.input_text),
                    token_places=cfg["token_places"],
                    take_promt_act=cfg["prompt_token"],
                    layers = cfg["layers"],
                    verbose=cfg["generation_verbose"],
                )

                if act_dict is None:
                    sys.stderr.write("No activation dictionary returned for the input: {}\n".format(data_elt.input_text))
                    data_elt.activations = None
                    continue
                # Put the gathered activation and output in the data element
                layer_list = [act_dict["hook"][layer_nb]["normalized"] for layer_nb in cfg["layers"]]
                data_elt.activations = layer_list
                data_elt.output_text = act_dict["output"]
                data_elt.input_token_length = act_dict["input_token_length"]
                data_elt.input_token_length = act_dict["prompt_token_length"]
                # data_elt.prompt_token_emb = act_dict["prompt_token_emb"]
                # data_elt.gen_token_emb = act_dict["gen_token_emb"]
                # data_elt.input_tokens = act_dict["input_tokens"]
                # data_elt.input_tokens_user = act_dict["input_tokens_user"]
                # data_elt.input_tokens_system = act_dict["input_tokens_system"]
                # data_elt.output_tokens = act_dict["output_tokens"]
                data_elt.input_tokens_str = act_dict["input_tokens_str"]
                data_elt.input_tokens_user_str = act_dict["input_tokens_user_str"]
                data_elt.input_tokens_system_str = act_dict["input_tokens_system_str"]
                data_elt.output_tokens_str = act_dict["output_tokens_str"]


                # Give an approximation of the storage size of the data element

            # Remove all the data_elt that haven't been populated
            prepared_type = [data_elt for data_elt in prepared_type if data_elt.activations is not None]

            # save as pickle file the prepared type data lists
            with open(output_file_name, "wb") as fp:
                pickle.dump(prepared_type, fp)
                fp.close()
        msg = "The Key type {}, with {} different entries is saved with inference\n"
        sys.stderr.write(msg.format(prepared_type[0].__class__.__name__, len(prepared_type)))

        # Clean up the prepared type
        del prepared_type

    del wllm


if __name__ == "__main__":
    main()
