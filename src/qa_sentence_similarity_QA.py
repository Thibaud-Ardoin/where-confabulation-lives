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
        cfg["output_folder"], 
        "steer_out_{}.pkl".format("_".join(cfg["evaluation_set"]))
    )
    with open(results_files, "rb") as file:
        results = pickle.load(file)

    cfg = cfg["evaluation"]

    # Load the expected out as dictionary
    with open(cfg["expected_output_file"], "r") as file:
        expected_output = yaml.safe_load(file)

    # Load the BERT_type model
    model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2


    for elmt_dict in results:
        # Create the corpus of Generated text, positive, negative and IDK answers
        corpus = [elmt_dict['output_text']] + list(elmt_dict['data_elt'].correct) + list(elmt_dict['data_elt'].wrong) + idk_corpus
        embeddings = model.encode(corpus)

        correct_id = list(range(1, 1+len(elmt_dict['data_elt'].correct)))
        wrong_id = list(range(1+len(elmt_dict['data_elt'].correct), 1+len(elmt_dict['data_elt'].correct)+len(elmt_dict['data_elt'].wrong)))
        idk_id = list(range(1+len(elmt_dict['data_elt'].correct)+len(elmt_dict['data_elt'].wrong), 
                        1+len(elmt_dict['data_elt'].correct)+len(elmt_dict['data_elt'].wrong)+len(idk_corpus)))
        
        # Calculate pairwise cosine similarity
        pairwise_similarity = model.similarity(embeddings, embeddings)

        correct_similarity = pairwise_similarity[0, correct_id]
        wrong_similarity = pairwise_similarity[0, wrong_id]
        idk_similarity = pairwise_similarity[0, idk_id]
        
        print(elmt_dict['output_text'])
        print(elmt_dict['data_elt'].correct)
        print(elmt_dict['data_elt'].wrong)
        # print(pairwise_similarity)
        print("Correct Answer Similarity (MiniLM):", correct_similarity)
        print("Wrong Answer Similarity (MiniLM):", wrong_similarity)
        print("IDK Answer Similarity (MiniLM):", idk_similarity)

        print(elmt_dict['output_text'])
        id_max_sim = t.argmax(pairwise_similarity[0, 1:]).item() +1

        print(correct_id, wrong_id, idk_id)
        print("id of max sim", id_max_sim, max(pairwise_similarity[0, 1:]))

        print("More similar too: ", corpus[id_max_sim], max(pairwise_similarity[0, 1:]))
        if id_max_sim in correct_id:
            evaluation = 1
        elif id_max_sim in wrong_id:
            evaluation = 0
        elif id_max_sim in idk_id:
            evaluation = 2
        else:
            evaluation = 3
        print(" >>> Would be evaluated as:", evaluation)

        meansim_correct = t.mean(correct_similarity)
        meansim_wrong = t.mean(wrong_similarity)
        meansim_idk = t.mean(idk_similarity)
        max_mean_sim = max([meansim_correct, meansim_wrong, meansim_idk])

        if meansim_correct > t.max(meansim_idk, meansim_wrong):
            mean_evaluation = 1
        elif meansim_wrong > t.max(meansim_correct, meansim_idk):
            mean_evaluation = 0
        elif meansim_idk > t.max(meansim_correct, meansim_wrong):
            mean_evaluation = 2
        else:
            mean_evaluation = 3
        print(" >>> As a mean of the similarities, it Would be evaluated as:", mean_evaluation)
        print()
        print()

        elmt_dict.update({
            "id_max_sim": id_max_sim, 
            "similarities": pairwise_similarity[0], 
            "evaluation": evaluation, 
            "mean_evaluation": mean_evaluation,
            "correct_id": correct_id,
            "wrong_id": wrong_id,
            "idk_id": idk_id,
            "max_mean_sim": max_mean_sim})

    # # Plot the pairwise similarity matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(pairwise_similarity, annot=True, cmap='coolwarm', cbar=True, xticklabels=corpus, yticklabels=corpus)
    # plt.title('Pairwise Cosine Similarity Matrix')
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()


# Save the evaluated results
with open(outfilename, "wb") as file:
    pickle.dump(results, file)




if __name__ == "__main__":
    main()