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


class Hook:
    def __init__(self):
        self.out = []

    def __call__(self, module, module_inputs, module_outputs):
        self.out.append(module_outputs)

class Hook_input:
    def __init__(self):
        self.input = None

    def __call__(self, module, module_inputs, module_outputs):
        self.input = module_inputs


def token2string(generator, tokens):
    """ 
        Utilise the Generator parts to hack the string value of a given token. 
        Could be pretty inaccurate compare to expected output, as it is sometimes intermmediate tokens
    """
    normalized = generator.model.norm(tokens)
    logits = generator.model.output(normalized.clone())
    probs = torch.softmax(logits / 0.5, dim=-1)
    next_token = sample_top_p(probs[0], 0.9)
    string_res = generator.tokenizer.decode([next_token])
    return string_res






def get_acts(system_prompt, statements, generator, layers, device, verbose=False):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = generator.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    input_hook = Hook_input()
    input_handle = generator.model.layers[0].register_forward_hook(input_hook)


    # get activations
    acts = {layer : [] for layer in layers}
    for statement in statements:
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": statement}, 
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
        
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out)

        # String conversion of the "input hook". Looks like distorted from the original input.
        strings1 = token2string(generator, input_hook.input[0])

        if verbose:
            print("input:", statement)
            print("Out content: ", results[0]['generation']['content'])

    
    # remove hooks
    for handle in handles:
        handle.remove()
    input_handle.remove()
    
    return acts, results

def gather_inference_dict(generator, sys_prompt, usr_prompt, token_places, take_promt_act, layers, verbose=False):
    # Get activation from forward pass
    acts, results = get_acts(sys_prompt, [usr_prompt], generator, layers, "CUDA", verbose=verbose)

    input_tokens = generator.tokenizer.encode(" ".join([sys_prompt, usr_prompt]), bos=True, eos=False)
    input_token_length = len(input_tokens) + 2

    # Adjust the selected token positions
    if token_places == "all":
        token_places = list(range(len(acts[layers[0]][0])))
    elif take_promt_act :
        token_places.append(0)

    data = {}
    data["output"] = results[0]['generation']['content']
    data["input_token_length"] = input_token_length
    data["prompt_token_emb"] = generator.model.norm(generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
    data["gen_token_emb"] = generator.model.norm(generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
    data["hook"] = {}
    with torch.no_grad():
        for layer_nb in layers:     # Loop through 32 layer at max
            data["hook"][layer_nb] = {}
            data["hook"][layer_nb]["normalized"] = []
            # data["hook"][layer_nb]["acts"] = []
            # data["hook"][layer_nb]["logits"] = []
            data["hook"][layer_nb]["tokens"] = []
            for i in token_places: # Loop through the desired tokens (or just on)
                # Raw activations gathered from the hooks
                act = acts[layer_nb][0][i][0]
                # data["hook"][layer_nb]["acts"].extend(act)
                if i==0 and take_promt_act: # Activation of the whole prompt
                    data["prompt_token_length"] = len(act)

                # Normalizing like it is done at the end of the model before logits
                # normalized = generator.model.norm(act)
                data["hook"][layer_nb]["normalized"].extend(act)#normalized)

                # Logits of the generation
                # logits = generator.model.output(normalized)
                # data["hook"][layer_nb]["logits"].extend(logits)

                # Next token prediction
                # probs = torch.softmax(logits / 0.5, dim=-1)
                # next_token = sample_top_p(probs, 0.9)
                # data["hook"][layer_nb]["tokens"].extend([generator.tokenizer.decode([n]) for n in next_token])

                # print("Layer {} Seq {}".format(layer_nb, i), [generator.tokenizer.decode([n]) for n in next_token])


    # Return the populated dictionary
    return data



def main():
    cfg = ConfigManager().config

    # Load the prepared pickle data
    prepared_data_list = []
    for input_type in cfg["experiment"]["data"]:
        prepared_data_list.append(pickle.load(open(os.path.join(cfg["prepare"]["prepared_data_folder"], input_type + ".pkl"), "rb")))

    cfg = cfg["inference"]
    torch.manual_seed(cfg["seed"])
    generator = Llama.build(
        ckpt_dir=cfg["model_path"],
        tokenizer_path=cfg["tokenizer_path"],
        max_seq_len=cfg["max_seq_len"],
        max_batch_size=cfg["max_batch_size"],
        seed = cfg["seed"]
    )


    sys.stderr.write("** Inference of the model! Can take a bit of time **\n")

    # Process inference of prompts and gather activations to populate the prepared data
    for prepared_type in prepared_data_list:
        for data_elt in prepared_type:
            act_dict = gather_inference_dict(generator, 
                                sys_prompt=data_elt.system_prompt,
                                usr_prompt=data_elt.user_prompt.format(data_elt.input_text),
                                token_places=cfg["token_places"],
                                take_promt_act=cfg["prompt_token"],
                                layers = cfg["layers"],
                                verbose=cfg["generation_verbose"],
            )

            # Put the gathered activation and output in the data element
            layer_list = [act_dict["hook"][layer_nb]["normalized"] for layer_nb in cfg["layers"]]
            data_elt.activations = layer_list
            data_elt.output_text = act_dict["output"]
            data_elt.input_token_length = act_dict["input_token_length"]
            data_elt.input_token_length = act_dict["prompt_token_length"]
            data_elt.prompt_token_emb = act_dict["prompt_token_emb"]
            data_elt.gen_token_emb = act_dict["gen_token_emb"]

        # save as pickle file the prepared type data lists
        with open(os.path.join(cfg["inference_data_folder"], "{}.pkl".format(data_elt.original_name)), "wb") as fp:
            pickle.dump(prepared_type, fp)
            fp.close()
        msg = "The Key type {}, with {} different entries is saved with inference\n"
        sys.stderr.write(msg.format(data_elt.__class__.__name__, len(prepared_type)))

    del generator


if __name__ == "__main__":
    main()
