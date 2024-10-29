import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import os
import yaml

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
    cfg = cfg["evaluation"]

    # Load the expected out as dictionary
    with open(cfg["expected_outputs_path"], "r") as file:
        expected_output = yaml.safe_load(file)[expected_type]

    print("expected_output", expected_output)

    # Load the BERT_type model
    model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2


    for elmt_dict in results:
        print(elmt_dict.output_text)
        print(expected_output)
        # Create the corpus of Generated text, positive and negative
        corpus = [elmt_dict.output_text] + list(expected_output['yes']) + list(expected_output['no'])
        embeddings = model.encode(corpus)

        correct_id = list(range(1, 1+len(expected_output['yes'])))
        wrong_id = list(range(1+len(expected_output['yes']), 1+len(expected_output['yes'])+len(expected_output['no'])))
        
        # Calculate pairwise cosine similarity
        pairwise_similarity = model.similarity(embeddings, embeddings)

        correct_similarity = pairwise_similarity[0, correct_id]
        wrong_similarity = pairwise_similarity[0, wrong_id]
        
        print(elmt_dict.output_text)
        print(expected_output['yes'])
        print(expected_output['no'])
        # print(pairwise_similarity)
        print("Correct Answer Similarity (MiniLM):", correct_similarity)
        print("Wrong Answer Similarity (MiniLM):", wrong_similarity)


        # Compile max similarities
        print(elmt_dict.output_text)
        id_max_sim = t.argmax(pairwise_similarity[0, 1:]).item() +1

        print(correct_id, wrong_id)
        print("id of max sim", id_max_sim, max(pairwise_similarity[0, 1:]))

        print("More similar too: ", corpus[id_max_sim], max(pairwise_similarity[0, 1:]))
        if id_max_sim in correct_id:
            evaluation = 1
        elif id_max_sim in wrong_id:
            evaluation = 0
        else:
            evaluation = 2

        # If the similarity is below a certain threshold, we consider it as UNDECIDED
        if max(pairwise_similarity[0, 1:]) < cfg["similarity_decision_threshold"]:
            evaluation = 2

        print(" >>> Would be evaluated as:", evaluation)

        # Compile mean similarities
        meansim_correct = t.mean(correct_similarity)
        meansim_wrong = t.mean(wrong_similarity)
        max_mean_sim = max([meansim_correct, meansim_wrong])

        if meansim_correct > meansim_wrong:
            mean_evaluation = 1
        elif meansim_wrong > meansim_correct:
            mean_evaluation = 0
        else:
            mean_evaluation = 2
        print(" >>> As a mean of the similarities, it Would be evaluated as:", mean_evaluation)
        print()

        elmt_dict.update({
            "id_max_sim": id_max_sim, 
            "similarities": pairwise_similarity[0], 
            "evaluation": evaluation, 
            "mean_evaluation": mean_evaluation,
            "correct_id": correct_id,
            "wrong_id": wrong_id,
            "max_mean_sim": max_mean_sim})

        # # Plot the pairwise similarity matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(pairwise_similarity, annot=True, cmap='coolwarm', cbar=True, xticklabels=corpus, yticklabels=corpus)
        # plt.title('Pairwise Cosine Similarity Matrix')
        # plt.xticks(rotation=90)
        # plt.yticks(rotation=0)
        # plt.tight_layout()
        # plt.show()

    # Save the results
    if not os.path.exists(cfg["output_folder"]):
        os.makedirs(cfg["output_folder"])
    out_file = os.path.join(
        cfg["output_folder"], 
        "steer_out_{}_evaluated.pkl".format("_".join(cfg["evaluation_set"]))
    )

    # Save the evaluated results
    with open(out_file, "wb") as file:
        pickle.dump(results, file)




if __name__ == "__main__":
    main()