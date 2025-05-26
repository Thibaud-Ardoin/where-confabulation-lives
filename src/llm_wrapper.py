from transformers import pipeline
import torch

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama


def create_llm_wrapper(**kwargs):
    model_name = kwargs.get("model_name")
    if model_name == "Meta-Llama-3-8B-Instruct":
        return LLMWrappedLLama3(**kwargs)
    elif model_name == "gpt2-small":
        return Gpt2Wrapper(**kwargs)
    elif model_name == "Gemma-2b":
        return GemmaWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class LLMWrapper:
    """
    A wrapper class for interacting with a pre-trained language model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LLMWrapper with a specified model and task.
        """
        self.model_name = kwargs["model_name"]
        self.cfg = kwargs["config"]
        
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



class LLMWrappedLLama3(LLMWrapper):
    """
    A wrapper class for the Llama 3 model.
    """
    def __init__(self, **kwargs):
        """
        Initialize the LLMWrappedLLama3 with a specified model and task.
        """
        super().__init__(**kwargs)
        self.model_name = kwargs["model_name"]
        self.cfg = kwargs["config"]
        self.load_model()
        print("Wrapper Initialisation completed")

    def load_model(self):

        print("Args from generator", 
                self.cfg["model_path"],
                self.cfg["tokenizer_path"],
                self.cfg["max_seq_len"],
                self.cfg["max_batch_size"],
                self.cfg["seed"]
        )

        self.generator = Llama.build(
            ckpt_dir=self.cfg["model_path"],
            tokenizer_path=self.cfg["tokenizer_path"],
            max_seq_len=self.cfg["max_seq_len"],
            max_batch_size=self.cfg["max_batch_size"],
            seed = self.cfg["seed"]
        )
        print("Llama model loaded !!", self.generator)

    def gather_inference_dict(
            self, 
            sys_prompt,
            usr_prompt,
            token_places,
            take_promt_act,
            layers,
            verbose):
            # Get activation from forward pass
        acts, results = self.get_acts(sys_prompt, [usr_prompt], layers, "CUDA", verbose=verbose)

        input_tokens = self.generator.tokenizer.encode(" ".join([sys_prompt, usr_prompt]), bos=True, eos=False)
        input_token_length = len(input_tokens) + 2

        data = {}
        data["output"] = results[0]['generation']['content']
        data["input_token_length"] = input_token_length
        # Tokens as ids
        data["input_tokens_user"] = input_tokens[len(sys_prompt):]
        data["input_tokens_system"] = input_tokens[:len(sys_prompt)]
        data["input_tokens"] = input_tokens
        data["output_tokens"] = self.generator.tokenizer.encode(results[0]['generation']['content'], bos=True, eos=False)

        # Tokens as strings
        data["input_tokens_user_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens_user"]]
        data["input_tokens_system_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens_system"]]
        data["input_tokens_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens"]]
        data["output_tokens_str"] = [self.generator.tokenizer.decode([t]) for t in data["output_tokens"]]

        data["prompt_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
        data["gen_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
        data["hook"] = {}

        # Adjust the selected token positions
        if token_places == "all":
            token_places = list(range(len(acts[layers[0]][0])))
        elif token_places == "first_gen":
            token_places = [1]
            data["prompt_token_length"] = 0
        elif take_promt_act :
            token_places.append(0)

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
    
    def get_acts(self, system_prompt, statements, layers, device, verbose=False):
        """
        Get given layer activations for the statements. 
        Return dictionary of stacked activations.
        """
        # attach hooks
        hooks, handles = [], []
        for layer in layers:
            hook = Hook()
            handle = self.generator.model.layers[layer].register_forward_hook(hook)
            hooks.append(hook), handles.append(handle)
        
        input_hook = Hook_input()
        input_handle = self.generator.model.layers[0].register_forward_hook(input_hook)


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
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=None,
                temperature=0.6,
                top_p=0.9,
                echo = False,
            )
            
            for layer, hook in zip(layers, hooks):
                acts[layer].append(hook.out)

            if verbose:
                print("input:", statement)
                print("Out content: ", results[0]['generation']['content'])

        
        # remove hooks
        for handle in handles:
            handle.remove()
        input_handle.remove()
        
        return acts, results


class GemmaWrapper(LLMWrapper):
    """
    A wrapper class for the Gemma model.
    """
    def __init__(self, **kwargs):
        """
        Initialize the GemmaWrapper with a specified model and task.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.cfg = config
        self.load_model()




class Gpt2Wrapper(LLMWrapper):
    """
    A wrapper class for the GPT-2 model.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Gpt2Wrapper with a specified model and task.
        """
        super().__init__(**kwargs)
        self.model_name = kwargs["model_name"]
        self.cfg = kwargs["config"]
        self.device = kwargs.get("device")
        self.hook_memorry = []
        self.load_model()

    def load_model(self):
        self.model = HookedTransformer.from_pretrained("gpt2-small", device=self.device)


    def memorry_hook(self, activation: torch.Tensor, hook: HookPoint, your_args=None) -> torch.Tensor:
        """
        activation: Tensor of shape [batch, position, d_model]
        hook: Contains metadata about hook location
        your_args: Custom arguments for your intervention
        """
        self.hook_memorry.append(activation)
        return activation


    def hook_activations(self, system_prompt, statements, layers, verbose=False):
        self.model.reset_hooks()

        # Compile the locations
        hook_locations = []
        for layer in layers:
            hook_location = "blocks.{}.hook_resid_post".format(layer)
            hook_locations.append(hook_location)

            # attach hooks
            self.model.add_hook(hook_location, self.memorry_hook)


        # get activations
        acts = []
        retsults = []
        for statement in statements:
            print("statement: ", statement)

            output = self.model.generate(
                statement,
                max_new_tokens=self.cfg["max_seq_len"],
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                do_sample=True
            )

            # stack the whole memorry for each fwd            
            acts.append(self.hook_memorry)

            retsults.append(output)
            # Reset the hook memory for the next statement
            self.hook_memorry = []

            if verbose:
                print("input:", statement)
                print("Out content: ", output)

        
        return acts, retsults

    def gather_inference_dict(
            self,
            sys_prompt,
            usr_prompt,
            token_places,
            take_promt_act,
            layers,
            verbose):
        
        activations, result = self.hook_activations(
            system_prompt=sys_prompt, 
            statements=[usr_prompt], 
            layers=layers, 
            verbose=verbose
        )

        print("Activations: ", len(activations))
        print(len(activations[0]))
        print("Activations: ", activations[0][0].shape)
        print("Result: ", result)


        data = {}
        data["output"] = result[0]
        data["input_token_length"] = input_token_length

        # Tokens as ids
        data["input_tokens_user"] = input_tokens[len(sys_prompt):]
        data["input_tokens_system"] = input_tokens[:len(sys_prompt)]
        data["input_tokens"] = input_tokens
        data["output_tokens"] = self.generator.tokenizer.encode(results[0]['generation']['content'], bos=True, eos=False)

        # Tokens as strings
        data["input_tokens_user_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens_user"]]
        data["input_tokens_system_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens_system"]]
        data["input_tokens_str"] = [self.generator.tokenizer.decode([t]) for t in data["input_tokens"]]
        data["output_tokens_str"] = [self.generator.tokenizer.decode([t]) for t in data["output_tokens"]]

        data["prompt_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
        data["gen_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
        data["hook"] = {}

        # Adjust the selected token positions
        if token_places == "all":
            token_places = list(range(len(acts[layers[0]][0])))
        elif token_places == "first_gen":
            token_places = [1]
            data["prompt_token_length"] = 0
        elif take_promt_act :
            token_places.append(0)

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
                    data["hook"][layer_nb]["normalized"].extend(act)#normalized)


        return data