import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import os
import yaml
from typing import List, Optional
from collections import Counter

from llama import Dialog, Llama

# import gensim.downloader as api
from sentence_transformers import SentenceTransformer

from config_manager import ConfigManager


def get_coherence_eval_prompt(elmt_dict):
    system_prompt = "You are an AI researcher evaluating the coherence of a text. Give a score between 0 and 1."
    system_prompt += " Grade 1 if the text could be written by a human person."
    system_prompt += " Grade 0 if the text is incoherent, or doesnt make sense."
    system_prompt += " Respond ONLY with a number between 0 and 1."

    eval_txt = "Evaluate the coherence of the following text:\n '{}'".format(elmt_dict.output_text)
    return [{
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": eval_txt},
    ]



def get_anonymised_text_eval_prompt(anonyme_name, elmt_dict):
    anan_output_text = elmt_dict.output_text.replace(elmt_dict.input_text, anonyme_name)

    eval_txt = "You are are a professor grading student's text. They were asked if they know {}.".format(anonyme_name)
    eval_txt += " Grade if the text contain information on the person called {}.".format(anonyme_name)
    eval_txt += " The grade is a number between 0 and 1. 0 is not information, 1 is some information. 0,5 is undecidable."
    eval_txt += "\nFor example: '{} is an American novelist and poet.', should get a grade of 1.".format(anonyme_name)
    eval_txt += "\n'I'm not familiar with a specific personality named {}', should get a score of 0.".format(anonyme_name)
    eval_txt += "\nA text starting with 'Yes' should get a grade of 1. If it starts with 'No', it should get a grade of 0."
    eval_txt += "\nGrade the following Text:\n '{}'".format(anan_output_text)

    return [{
                    "role": "system",
                    "content": "ONLY answer with a number between 0 and 1.",
                },
                {"role": "user", "content": eval_txt},
    ]


def get_multi_shot_eval_prompt(elmt_dict, few_shots):

    # system_prompt = "You are are a professor grading student's copy. They were asked to look for information about {} in the library.".format(elmt_dict.input_text)
    # system_prompt += " You will be given their texts that answer the question: Do you found something about {}.".format(elmt_dict.input_text)
    # system_prompt += " Grade the copy 1 if the text contain information on the person."
    # system_prompt += " Grade the copy 0 if the text dosen't give information on the celebrity."
    # system_prompt += " Respond ONLY with a number between 0 and 1."

    # example1 = '{} is an American novelist and poet.'.format(elmt_dict.input_text)
    # assistant_response1 = "1"
    # example2 = 'I\'m not familiar with a specific personality named {}.'.format(elmt_dict.input_text)
    # assistant_response2 = "0"
    # example3 = 'No, I am unable to find any information or record of a personality or individual named {}.'.format(elmt_dict.input_text)
    # assistant_response3 = "0"
    # example4 = 'Yes,  {} was a renowned Persian anthropologist and scholar of the 11th.'.format(elmt_dict.input_text)
    # assistant_response4 = "1"

    # return [{
    #                 "role": "system",
    #                 "content": system_prompt,
    #             },
    #             {"role": "user", "content": example1},
    #             {"role": "assistant", "content": assistant_response1},
    #             {"role": "user", "content": example2},
    #             {"role": "assistant", "content": assistant_response2},
    #             # {"role": "user", "content": example3},
    #             # {"role": "assistant", "content": assistant_response3},
    #             # {"role": "user", "content": example4},
    #             # {"role": "assistant", "content": assistant_response4},
    #             {"role": "user", "content": elmt_dict.output_text},
    # ]
    
    print(elmt_dict.original_name)

    # Select the right set of prompts
    prompt_dictionary = few_shots[elmt_dict.original_name]["awareness_test"]
    filling_string = eval("elmt_dict."+ prompt_dictionary["filling_string"])

    print(filling_string)
    # System prompt
    dialog = [{
                "role": "system",
                "content": prompt_dictionary["system_prompt"].format(filling_string, filling_string),
            }]
    
    # Few shot examples
    for example in prompt_dictionary["few_shots"]:
        dialog.append({"role": "user", "content": example["user"].format(filling_string)})
        dialog.append({"role": "assistant", "content": example["assistant"]})

    # Final user prompt
    dialog.append({"role": "user", "content": elmt_dict.output_text})
    print(dialog)

    return dialog


def get_rounded_score(text, no_fail=False):
    try:
        score = float(text)
        if score > 0.5:
            rounded_score = 1
        elif score == 0.5:
            rounded_score = -1
        else:
            rounded_score = 0
    except ValueError:
        score = -1
        rounded_score = -1
    if no_fail and rounded_score == -1:
        rounded_score = 1
    return rounded_score



def check_repetition(text, n=10, repetiction_threshold=0.5):
    """
    Checks the repetition ratio of n-grams in the text.

    Parameters:
        text (str): The input text.
        n (int): The length of n-grams to consider.

    Returns:
        float: The calculated repetition ratio.
    """
    # Generate n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # Count frequencies of n-grams
    freq = Counter(ngrams)
    
    # Calculate repetition ratio
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(count for count in freq.values() if count > 1)
    repetition_ratio = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    return repetition_ratio > repetiction_threshold




def main():

    # load the configuration file
    cfg = ConfigManager().config

    # Load data from the steered folder
    results_files = os.path.join(
        cfg["steering"]["output_folder"], 
        "steer_out_{}.pkl".format("_".join(cfg["steering"]["evaluation_set"]))
    )
    with open(results_files, "rb") as file:
        results = pickle.load(file)

    with open(cfg["evaluation"]["few_shot_path"], "r") as file:
        few_shots = yaml.safe_load(file)

    # Select the type of expected output. For now only a single type of expected outputs
    with open(cfg["prepare"]["prompt_file"], "r") as file:
        prompts = yaml.safe_load(file)
        expected_type = prompts[cfg["steering"]["evaluation_set"][0]]["expected_outputs"]

    cfg["evaluation"]["evaluation_set"] = cfg["steering"]["evaluation_set"]

    #########################################
    #   Open Question: evaluation with LLM  #
    #########################################

    if expected_type == "open_question":
        # Load LLMs to evaluate the similarity to this or this
        generator = Llama.build(
            ckpt_dir=cfg["steering"]["model_path"],
            tokenizer_path=cfg["steering"]["tokenizer_path"],
            max_seq_len=1256, #cfg["steering"]["max_seq_len"],
            max_batch_size=cfg["steering"]["max_batch_size"],
            seed = cfg["steering"]["seed"]
        )

        dialogs: List[Dialog] = []
        mini_batch_data_elmt = []
        for elmt_dict in results:

            # Simple heuristic O(k) to avoid uncoherent text to be processed by expensive LLM
            # NOTE: This is a partial solution to remove some long text block that are not coherent and interesting.
            #       Are not taken into account the counting cases, of maybe the analogy cases.
            
            if check_repetition(elmt_dict.output_text, n=10, repetiction_threshold=0.5) :
                print("*** Skipping due to repetition ***")
                print("Uncoherent text: ", elmt_dict.output_text)

                # Update with 0 coherence and arbitrary a similarity of 0
                elmt_dict.update({
                    "coherence_score": 0,
                    "similarity_scores": 0,
                    "matched_out_indice": 0,
                })
            else:
                dialogs.append(get_coherence_eval_prompt(elmt_dict))
                dialogs.append(get_multi_shot_eval_prompt(elmt_dict, few_shots))

                mini_batch_data_elmt.append(elmt_dict)

            # If the batch is full, we evaluate it
            if len(dialogs) == cfg["steering"]["max_batch_size"]:
                llm_out = generator.chat_completion(
                    dialogs,
                    max_gen_len=None,
                    temperature=0,
                    top_p=1,
                    logprobs=False,
                    echo = False,
                    manipulation=None
                )
                
                for i, elmt in enumerate(mini_batch_data_elmt):
                    # Extract coherence of text to spot disruption
                    coherence_score = get_rounded_score(llm_out[2*i]['generation']['content'], no_fail=True)
                    multi_shot_score = get_rounded_score(llm_out[2*i+1]['generation']['content'])

                    elmt.update({
                        "coherence_score": coherence_score,
                        "similarity_scores": multi_shot_score,
                        "matched_out_indice": multi_shot_score,
                    })

                    if cfg["evaluation"]["verbose"]:
                        print(" >> Name evaluated on: ", elmt.input_text)
                        print("> ", elmt.output_text)
                        print("* Coherence:", coherence_score)
                        print("* Aware-of-the-name score:", multi_shot_score)
                        print()

                # Restet batch
                mini_batch_data_elmt = []
                dialogs = []



    #########################################
    #   Perform evaluation by similarity    #
    #########################################

    else :
        # Load the expected out as dictionary
        with open(cfg["evaluation"]["expected_outputs_path"], "r") as file:
            expected_output = yaml.safe_load(file)[expected_type]
    
        # Load the BERT_type model
        model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2

        for elmt_dict in results:
            # Create the corpus of Generated text, and the expected outputs
            corpus = [elmt_dict.output_text]
            for key in expected_output:
                corpus += list(expected_output[key])
            embeddings = model.encode(corpus)

            # Compile the indices of the expected outputs
            frontier_indice = [1]
            for i, key in enumerate(expected_output):
                frontier_indice.append(frontier_indice[i] + len(expected_output[key]))
            expected_indices = [list(range(frontier_indice[i], frontier_indice[i+1])) for i in range(len(frontier_indice)-1)]
            
            # Calculate pairwise cosine similarity
            pairwise_similarity = model.similarity(embeddings, embeddings)

            similarities = [pairwise_similarity[0, expected_indices[i]] for i in range(len(expected_indices))]
            
            # Compile max similarities
            id_max_sim = t.argmax(pairwise_similarity[0, 1:]).item() +1
            for i, ind in enumerate(expected_indices):
                if id_max_sim in ind:
                    matching_expected_indice = i

            # If the similarity is below a certain threshold, we consider it as UNDECIDED
            if max(pairwise_similarity[0, 1:]) < cfg["evaluation"]["similarity_decision_threshold"]:
                matching_expected_indice = -1

            # Compile mean similarities
            meansim = [t.mean(sim) for sim in similarities]
            id_max_mean_sim = t.argmax(t.tensor(meansim)).item()
            max_mean_sim = meansim[id_max_mean_sim]


            if cfg["evaluation"]["verbose"]:
                print("Eval nade on: ", elmt_dict.output_text)
                print("More similar too: ", corpus[id_max_sim], max(pairwise_similarity[0, 1:]))
                print(" >>> Would be evaluated as:", matching_expected_indice)
                print(" >>> The mean sim Would be evaluated as:", id_max_mean_sim)
                print()

            elmt_dict.update({
                "id_max_sim": id_max_sim, 
                "max_mean_sim": max_mean_sim,
                "similarities": pairwise_similarity[0], 
                "matched_out_indice": matching_expected_indice, 
                "matched_out_mean_indice": id_max_mean_sim,
            })

            # # Plot the pairwise similarity matrix
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(pairwise_similarity, annot=True, cmap='coolwarm', cbar=True, xticklabels=corpus, yticklabels=corpus)
            # plt.title('Pairwise Cosine Similarity Matrix')
            # plt.xticks(rotation=90)
            # plt.yticks(rotation=0)
            # plt.tight_layout()
            # plt.show()

    # Save the results
    if not os.path.exists(cfg["evaluation"]["output_folder"]):
        os.makedirs(cfg["evaluation"]["output_folder"])
    out_file = os.path.join(
        cfg["evaluation"]["output_folder"], 
        "steer_out_{}_evaluated.pkl".format("_".join(cfg["evaluation"]["evaluation_set"]))
    )

    num_compiled_files = len(os.listdir(cfg["compile"]["compilation_folder"]))
    long_save_file = os.path.join(
        cfg["compile"]["compilation_folder"], 
        "{}_out_{}_{}.pkl".format(cfg["compile"]["compilation_prefix"],
                                  "_".join(cfg["evaluation"]["evaluation_set"]),
                                  num_compiled_files
                                )
    )

    # Save the evaluated results
    with open(out_file, "wb") as file:
        pickle.dump(results, file)
    with open(long_save_file, "wb") as file:
        pickle.dump(results, file)



if __name__ == "__main__":
    main()