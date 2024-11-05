import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import os
import yaml
from typing import List, Optional

from llama import Dialog, Llama

# import gensim.downloader as api
from sentence_transformers import SentenceTransformer

from config_manager import ConfigManager


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
            max_seq_len=256, #cfg["steering"]["max_seq_len"],
            max_batch_size=cfg["steering"]["max_batch_size"],
            seed = cfg["steering"]["seed"]
        )

        for elmt_dict in results:

            # Trying to anonymise the text:
            anonyme_name = "John Doe"
            anan_output_text = elmt_dict.output_text.replace(elmt_dict.input_text, anonyme_name)

            eval_txt = "You are are a professor grading student's text. They were asked if they know {}.".format(anonyme_name)
            eval_txt += " Grade if the text contain information on the person called {}.".format(anonyme_name)
            eval_txt += " The grade is a number between 0 and 1. 0 is not information, 1 is some information. 0,5 is undecidable."
            eval_txt += "\nFor example: '{} is an American novelist and poet.', should get a grade of 1.".format(anonyme_name)
            eval_txt += "\n'I'm not familiar with a specific personality named {}', should get a score of 0.".format(anonyme_name)
            eval_txt += "\nA text starting with 'Yes' should get a grade of 1. If it starts with 'No', it should get a grade of 0."
            eval_txt += "\nGrade the following Text:\n '{}'".format(anan_output_text)

            # Create Dialog from data
            dialogs: List[Dialog] = [[
                {
                    "role": "system",
                    "content": "ONLY answer with a number between 0 and 1.",
                },
                {"role": "user", "content": eval_txt}, 
            ]]

            # Create Response from model
            # try:
            llm_out = generator.chat_completion(
                dialogs,
                max_gen_len=None,
                temperature=0,
                top_p=1,
                logprobs=False,
                echo = False,
                manipulation=None
            )
            # except:
            #     results = [{"generation": {"content": "Error"}}]


            print('<"', llm_out[0]['generation']['content'])

            try:
                similarity_score = float(llm_out[0]['generation']['content'])
                if similarity_score > 0.5:
                    matching_expected_indice = 1
                elif similarity_score == 0.5:
                    matching_expected_indice = -1
                else:
                    matching_expected_indice = 0
            except ValueError:
                similarity_score = -1
                matching_expected_indice = -1

            print("similarity_score and rounded up", similarity_score, matching_expected_indice)


            elmt_dict.update({
                "similarity_score": similarity_score,
                "matched_out_indice": matching_expected_indice,
            })

            print(elmt_dict)


            if cfg["evaluation"]["verbose"]:
                print(" >> Name evaluated on: ", eval_txt)
                print("> similarity_score:", similarity_score)
                print()



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

    # Save the evaluated results
    with open(out_file, "wb") as file:
        pickle.dump(results, file)




if __name__ == "__main__":
    main()