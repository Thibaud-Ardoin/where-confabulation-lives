from transformers import pipeline
import torch
import numpy as np

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import transformers


import pandas as pd
from typing import List, Optional
from llama import Dialog, Llama


def create_llm_wrapper(**kwargs):
    model_name = kwargs.get("model_name")
    if model_name == "Meta-Llama-3-8B-Instruct":
        return LLMWrappedLLama3(**kwargs)
    # elif model_name == "gpt2-small":
    #     return Gpt2Wrapper(**kwargs)
    # elif model_name == "Gemma-2b":
    #     return GemmaWrapper(**kwargs)
    # elif model_name == "Gemma-9b":
    #     return GemmaWrapper(**kwargs)
    else:
        return HuggingFaceWrapper(**kwargs)
        # raise ValueError(f"Unknown model: {model_name}")


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

    def load_model(self):
        self.generator = Llama.build(
            ckpt_dir=self.cfg["model_path"],
            tokenizer_path=self.cfg["tokenizer_path"],
            max_seq_len=self.cfg["max_seq_len"],
            max_batch_size=self.cfg["max_batch_size"],
            seed = self.cfg["seed"]
        )

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
    


############################
##  Hugging Face Wrapper  ##
############################


class HuggingFaceWrapper(LLMWrapper):
    """
    A wrapper class for the Hugging face models.
    """
    def __init__(self, **kwargs):
        """
        Initialize the with a specified model and task.
        """
        super().__init__(**kwargs)
        self.model_name = kwargs["model_name"]
        self.cfg = kwargs["config"]
        self.device = kwargs.get("device")
        self.hook_memorry = {}
        self.arguments = kwargs
        print("self.arguments", self.arguments)
        self.pipeline = self.load_model(self.arguments)
        self.tokenizer = self.pipeline.tokenizer
        self.active_hooks = []

    def load_model(self, model_name: str):
        """
            Load the model from Hugging Face Transformers library.
        """
        if self.model_name == "Qwen2.5-7B-Instruct":
            loading_name = "Qwen/Qwen2.5-7B-Instruct"
        elif self.model_name == "Qwen2.5-14B-Instruct":
            loading_name = "Qwen/Qwen2.5-14B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation", 
            model=loading_name, 
            model_kwargs={"torch_dtype": eval(self.arguments["config"]["torch_dtype"])}, 
            device_map=None,  # disables offloading
            # token=self.arguments["hf_token"]
        )
        return pipeline
    
    def __call__(self, input, **kwargs):
        """
            Call the model with input text and additional arguments.
            
            Args:
                input_text (str): Input text to generate output from the model.
                **kwargs: Additional arguments for the model.
                
            Returns:
                str: Generated text from the model.
        """
        return self.pipeline.model(input, **kwargs)



    def register_hooks(self, hook_type, layer_index, steering_vector=None):
        """
            Register hooks to the model for gathering activations or steering outputs.
            
            Args:
                hook_type (str): Type of hook to register, either "gather" or "steering".
                layer_index (list): List of layer indices to register hooks on.
                steering_vector (list, optional): Vector to steer the output if hook_type is "steering".
                
            Returns:
                list: List of registered hooks.
        """
        hooks = []
        if hook_type == "gather":
            for layer in layer_index:
                activation_memorry = {}
                hook = self.pipeline.model.model.layers[layer].register_forward_hook(
                    lambda module, input, output, layer_index=layer: hook_act_gather(module, input, output, layer_index, activation_memorry)
                )
                hook.activation_memorry = activation_memorry
                hooks.append(hook)
        elif hook_type == "steering":
            for layer in layer_index:
                hook = self.pipeline.model.model.layers[layer].register_forward_hook(
                    lambda module, input, output, layer_index=layer: hook_act_steering(module, input, output, layer_index, key_vector=steering_vector)
                )
                hooks.append(hook)
        return hooks
    
    def hook_activations(self, system_prompt, statements, layers, verbose=False):
        generation_arguments = {
            "do_sample": True,
            "temperature": self.arguments["config"]["temperature"],
            "max_new_tokens": self.arguments["config"]["max_seq_len"],
            "top_p": self.arguments["config"]["top_p"],
            "pad_token_id": None, 
            "eos_token_id": None,
        }

        detection_data = []
        # saved_activation = {}

        saving_hooks = self.register_hooks("gather", layers)


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
            # Run generation
            results = self.pipeline(
                dialogs,
                return_tensors=True,
                **generation_arguments
            )

            ids = results[0][0]["generated_token_ids"]        # torch.Tensor of token IDs
            tokens = self.tokenizer.convert_ids_to_tokens(ids)    # human‑readable tokens

            # For Qwen2.5-7B-Instruct
            imstart_token_id, assistante_token = 151644, 77091
            # find the place in the text where the assistant starts
            imstart_indexes = [i for i, x in enumerate(ids) if x == imstart_token_id]
            imstart_index = imstart_indexes[-1]

            # Assert that the assistant token is not the first token
            assert ids[imstart_index+1] == assistante_token

            output_ids = ids[imstart_index+2:]  # The output starts after the imstart_token_id
            output_text   = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            if verbose:
                print("Out content: ", output_text) #capturing the 3rd element corresponding to agent response

            # gather in order the activations memorries from the hooks
            activations = {}
            for hook in saving_hooks:
                for layer in layers:
                    if layer in hook.activation_memorry:
                        activations[layer] = hook.activation_memorry[layer].copy()
                        hook.activation_memorry[layer] = {"hooked_output": []}  # Clear the memory after gathering

            detection_data.append({
                "input": statement,
                "input_tokens_length": imstart_index - 1,
                "first_out_id": imstart_index + 1,
                "tokens_str": tokens,
                "token_ids": ids,
                "output_text": output_text,
                "activations": {layer: [act.type(torch.float16).detach().cpu().numpy() for act in activations[layer]["hooked_output"][0][0]] for layer in layers}
            })


        # Always remove hooks after you're done
        for hook in saving_hooks:
            hook.remove()

        return detection_data

    def gather_inference_dict(
            self,
            sys_prompt,
            usr_prompt,
            token_places,
            take_promt_act,
            layers,
            verbose):
        
        
        detection_data = self.hook_activations(
            system_prompt=sys_prompt, 
            statements=[usr_prompt], 
            layers=layers, 
            verbose=verbose
        )[0]  # Get the first result, as we only passed one statement

        # Shape of detection_data["activations"]:
        # dict(layer), list(seq_length), torch[d_model]

        # input_tokens = self.tokenizer.encode(" ".join([sys_prompt, usr_prompt]))
        # input_token_length = len(input_tokens) + 2  # +2 for bos and eos tokens

        data = {}
        data["output"] = detection_data["output_text"]
        data["input_token_length"] = detection_data["input_tokens_length"]

        # Tokens as ids
        # data["input_tokens_user"] = input_tokens[len(sys_prompt):]
        # data["input_tokens_system"] = input_tokens[:len(sys_prompt)]
        data["input_tokens"] = detection_data["token_ids"]
        data["output_tokens"] = detection_data["token_ids"][detection_data["first_out_id"]+1:]

        # Tokens as strings
        # TODO: Fix the tokenization to match the original input
        data["input_tokens_user_str"] = [detection_data["tokens_str"][:detection_data["input_tokens_length"]]]
        data["input_tokens_system_str"] = [detection_data["tokens_str"][:detection_data["input_tokens_length"]]]
        data["input_tokens_str"] = [detection_data["tokens_str"][:detection_data["input_tokens_length"]]]
        data["output_tokens_str"] = [detection_data["tokens_str"][detection_data["first_out_id"]+1:]]

        # data["prompt_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
        # data["gen_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
        data["hook"] = {}

        # Adjust the selected token positions
        if token_places == "all":
            token_places = list(range(len(detection_data["activations"][layers[0]])))
        elif token_places == "first_gen":
            data["prompt_token_length"] = 0
            token_places = [detection_data["first_out_id"]]

        no_generation = False

        # print("Output text:", detection_data["output_text"])
        # print("token place:", token_places)

        # print("total token list length:", len(detection_data["tokens_str"]))
        # print("Total tokens:", len(detection_data["activations"][0]))

        # print("input token length:", data["input_token_length"])
        # print("Output token length:", len(data["output_tokens"]))

        # print("input tokens user:", detection_data["input_tokens_length"])
        # print("first out id:", detection_data["first_out_id"])
        # print("this first token upstream is: ",detection_data["tokens_str"][detection_data["first_out_id"]])

        # print("activation layer 0 token place 0:", detection_data["activations"][0][0])
        # print("activation layer 0 token place 1:", detection_data["activations"][0][1])
        # print("activation layer 0 token place -1:", detection_data["activations"][0][-1])

        with torch.no_grad():
            # print("len(activations[0]) (layer id=0)", len(detection_data["activations"][0]))
            for nl, layer_nb in enumerate(layers):     # Loop through 32 layer at max
                data["hook"][layer_nb] = {}
                data["hook"][layer_nb]["normalized"] = []
                data["hook"][layer_nb]["tokens"] = []

                # print("len(activations[nl])", len(activations[nl]))

                for i in token_places: # Loop through the desired tokens (or just on)                  

                    # Raw activations gathered from the hooks
                    try:
                        act = np.array([detection_data["activations"][layer_nb][i]])
                    except IndexError as e:
                        print("IndexError: ", e)
                        print("nl: ", layer_nb, "i: ", i, "len(activations[nl]): ", len(detection_data["activations"][layer_nb]))
                        no_generation = True
                        act = torch.zeros((1, 2048))
                    
                    if i==0 and take_promt_act: # Activation of the whole prompt
                        data["prompt_token_length"] = len(act)

                    # print("act: ", act)
                    # Stack the activations
                    data["hook"][layer_nb]["normalized"].extend(act)

        if no_generation:
            return None

        return data
    
    def register_steering_hook(self, manipulation):
        steering_vector = torch.Tensor(manipulation["vector"]).to(self.pipeline.model.model.device)
        hooks = []
        for layer in manipulation["layers"]:
            hook = self.pipeline.model.model.layers[layer].register_forward_hook(
                lambda module, input, output, layer_index=layer: hook_act_steering(module, input, output, layer_index, key_vector=steering_vector)
            )
            hooks.append(hook)
            self.active_hooks.append(hook)
        return hooks
    
    def clear_hooks(self):
        """
        Clear all registered hooks.
        """
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks = []
    
    def text_generation(self, dialogs, max_gen_len, temperature, top_p, ):
        """
        Generate text using the model with the provided dialogs and parameters.
        
        Args:
            dialogs (List[Dialog]): List of Dialog objects containing the conversation history.
            max_gen_len (int): Maximum length of the generated text.
            temperature (float): Temperature for sampling.
            top_p (float): Top-p value for nucleus sampling.
        
        Returns:
            List[Dict]: List of dictionaries containing the generated text and other metadata.
        """
        return self.pipeline(
            dialogs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    







#####################
#   Hook Functions  #
#####################

def hook_act_gather(module, input, output, layer_index, activation_memorry):
    if layer_index not in activation_memorry:
        activation_memorry[layer_index] = {"hooked_output": []}
    activation_memorry[layer_index]["hooked_output"].extend(output)

def hook_act_steering(module, input, output, layer_index, key_vector=None):
    for i in range(len(output[0][0])):
        output[0][0][i] = output[0][0][i] + key_vector



















########################
#   LLM Lens Wrapper   #
########################




class GemmaWrapper(LLMWrapper):
    """
    A wrapper class for the Gemma familly.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Gpt2Wrapper with a specified model and task.
        """
        super().__init__(**kwargs)
        self.model_name = kwargs["model_name"]
        self.cfg = kwargs["config"]
        self.device = kwargs.get("device")
        self.hook_memorry = {}
        self.load_model()

    def load_model(self):
        if self.model_name == "Gemma-2b":
            self.model = HookedTransformer.from_pretrained("gemma-2b-it", device=self.device)
        # 17 layers, 2048 dim, 

        elif self.model_name == "Gemma-9b":
            from_pretrained_kwargs = {
                "torch_dtype": torch.bfloat16
            }
            self.model = HookedTransformer.from_pretrained(
                # "mistralai/Mistral-7B-Instruct-v0.1", 
                "Qwen/Qwen2.5-7B-Instruct",
                device=self.device,
                **from_pretrained_kwargs
            )
        # 32 layers, 4096 dim,


    def memorry_hook(self, activation: torch.Tensor, hook: HookPoint, your_args=None) -> torch.Tensor:
        """
        activation: Tensor of shape [batch, position, d_model]
        hook: Contains metadata about hook location
        your_args: Custom arguments for your intervention
        """
        layer_id = hook.name.split(".")[1]
        if layer_id not in self.hook_memorry:
            self.hook_memorry[layer_id] = []
        self.hook_memorry[layer_id].append(activation)

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

            # Need to compile the text input with special tokens and system prompt
            input_text = "\n".join([system_prompt, statement])

            output = self.model.generate(
                input_text,
                # system_prompt=system_prompt,
                max_new_tokens=self.cfg["max_seq_len"],
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                do_sample=True
            )

            # Shape is dict(layer), list(seq_length), torch[batch, token, d_model]
            layer_wise_act = []
            for layer in layers:
                # print("len hook memorry", len(self.hook_memorry[str(layer)]))    
                # print(self.hook_memorry[str(layer)][1].shape)
                layer_wise_act.append(self.hook_memorry[str(layer)])

            retsults.append(output)
            acts.append(layer_wise_act)

            # Reset the hook memory for the next statement
            self.hook_memorry = {}

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

        result = result[0]  # Get the first result, as we only passed one statement
        activations = activations[0]  # Get the first set of activations, as we only passed one statement
        
        # Shape is number_inputs(1), dict(layer), list(seq_length), torch[batch, token, d_model]
        # print("Activations: ", len(activations))
        # print(len(activations[0]))
        # print("Activations: ", activations[0][0][0].shape)
        # print("Result: ", result)

        # Actually there is no system prompt in that case..
        input_tokens = self.model.tokenizer.encode(usr_prompt) #" ".join([sys_prompt, usr_prompt]))
        # print("Input tokens: ", input_tokens)
        input_token_length = len(input_tokens) + 2  # +2 for bos and eos tokens

        data = {}
        data["output"] = result
        data["input_token_length"] = input_token_length

        # Tokens as ids
        data["input_tokens_user"] = input_tokens[len(sys_prompt):]
        data["input_tokens_system"] = input_tokens[:len(sys_prompt)]
        data["input_tokens"] = input_tokens
        data["output_tokens"] = self.model.tokenizer.encode(result)

        # Tokens as strings
        data["input_tokens_user_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens_user"]]
        data["input_tokens_system_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens_system"]]
        data["input_tokens_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens"]]
        data["output_tokens_str"] = [self.model.tokenizer.decode([t]) for t in data["output_tokens"]]

        # data["prompt_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
        # data["gen_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
        data["hook"] = {}

        # Adjust the selected token positions
        if token_places == "all":
            token_places = list(range(len(activations[layers[0]])))
        elif token_places == "first_gen":
            token_places = [1]
            data["prompt_token_length"] = 0
        elif take_promt_act :
            token_places.append(0)

        no_generation = False

        # for key in data:
        #     print("key", key)
        #     if isinstance(data[key], list):
        #         print("len data[key]", len(data[key]))
        #     print("data[key]", data[key])

        with torch.no_grad():
            print("len(activations[0]) (layer id=0)", len(activations[0]))
            for nl, layer_nb in enumerate(layers):     # Loop through 32 layer at max
                data["hook"][layer_nb] = {}
                data["hook"][layer_nb]["normalized"] = []
                data["hook"][layer_nb]["tokens"] = []

                # print("len(activations[nl])", len(activations[nl]))

                for i in token_places: # Loop through the desired tokens (or just on)                    
                    # print("len(activations[nl][i])", len(activations[nl][i]))
                    # print("len(activations[nl][i][0])", len(activations[nl][i][0]))
                    # print("len(activations[nl][i][0][0])", len(activations[nl][i][0][0]))

                    # Raw activations gathered from the hooks
                    try:
                        act = activations[nl][i][0]
                    except IndexError as e:
                        print("IndexError: ", e)
                        print("nl: ", nl, "i: ", i, "len(activations[nl]): ", len(activations[nl]))
                        no_generation = True
                        act = torch.zeros((1, 2048))
                    # print("Layer {} Seq {}".format(layer_nb, i), act.shape)
                    
                    if i==0 and take_promt_act: # Activation of the whole prompt
                        data["prompt_token_length"] = len(act)
                    # Stack the activations

                    data["hook"][layer_nb]["normalized"].extend(act)

        if no_generation:
            return None

        return data


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
        self.hook_memorry = {}
        self.load_model()

    def load_model(self):
        self.model = HookedTransformer.from_pretrained("gpt2-small", device=self.device)


    def memorry_hook(self, activation: torch.Tensor, hook: HookPoint, your_args=None) -> torch.Tensor:
        """
        activation: Tensor of shape [batch, position, d_model]
        hook: Contains metadata about hook location
        your_args: Custom arguments for your intervention
        """
        layer_id = hook.name.split(".")[1]
        if layer_id not in self.hook_memorry:
            self.hook_memorry[layer_id] = []
        self.hook_memorry[layer_id].append(activation)

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

            # Shape is dict(layer), list(seq_length), torch[batch, token, d_model]
            layer_wise_act = []
            for layer in layers:
                print("len hook memorry", len(self.hook_memorry[str(layer)]))    
                print(self.hook_memorry[str(layer)][0].shape)
                layer_wise_act.append(self.hook_memorry[str(layer)])

            retsults.append(output)
            acts.append(layer_wise_act)

            # Reset the hook memory for the next statement
            self.hook_memorry = {}

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

        result = result[0]  # Get the first result, as we only passed one statement
        activations = activations[0]  # Get the first set of activations, as we only passed one statement
        
        # Shape is number_inputs(1), dict(layer), list(seq_length), torch[batch, token, d_model]
        print("Activations: ", len(activations))
        print(len(activations[0]))
        print("Activations: ", activations[0][0][0].shape)
        print("Result: ", result)

        # Actually there is no system prompt in that case..
        input_tokens = self.model.tokenizer.encode(usr_prompt) #" ".join([sys_prompt, usr_prompt]))
        print("Input tokens: ", input_tokens)
        input_token_length = len(input_tokens) + 2  # +2 for bos and eos tokens

        data = {}
        data["output"] = result
        data["input_token_length"] = input_token_length

        # Tokens as ids
        data["input_tokens_user"] = input_tokens[len(sys_prompt):]
        data["input_tokens_system"] = input_tokens[:len(sys_prompt)]
        data["input_tokens"] = input_tokens
        data["output_tokens"] = self.model.tokenizer.encode(result)

        # Tokens as strings
        data["input_tokens_user_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens_user"]]
        data["input_tokens_system_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens_system"]]
        data["input_tokens_str"] = [self.model.tokenizer.decode([t]) for t in data["input_tokens"]]
        data["output_tokens_str"] = [self.model.tokenizer.decode([t]) for t in data["output_tokens"]]

        # data["prompt_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['prompt_token_id'])))
        # data["gen_token_emb"] = self.generator.model.norm(self.generator.model.tok_embeddings(torch.tensor(results[0]['gen_token_id'])))
        data["hook"] = {}

        # Adjust the selected token positions
        if token_places == "all":
            token_places = list(range(len(activations[layers[0]])))
        elif token_places == "first_gen":
            token_places = [0]
            data["prompt_token_length"] = 0
        elif take_promt_act :
            token_places.append(0)

        with torch.no_grad():
            for nl, layer_nb in enumerate(layers):     # Loop through 32 layer at max
                data["hook"][layer_nb] = {}
                data["hook"][layer_nb]["normalized"] = []
                data["hook"][layer_nb]["tokens"] = []

                for i in token_places: # Loop through the desired tokens (or just on)                    
                    # Raw activations gathered from the hooks
                    act = activations[nl][0][i][0]
                    
                    if i==0 and take_promt_act: # Activation of the whole prompt
                        data["prompt_token_length"] = len(act)
                    data["hook"][layer_nb]["normalized"].extend(act)


        return data