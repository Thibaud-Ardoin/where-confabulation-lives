import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go

from config_manager import ConfigManager




def plot_steering_wight_and_binary_out(list_result, cfg):
    """
    Plot the steering weight and accuracy for different clip values
    TODO: change the script to make it more general to every pair, trio of labels
    """

    # Convert list_result to a pandas DataFrame
    df = pd.DataFrame([elmt.__dict__ for elmt in list_result]) #__dict__ 

    # Filter out the undecisive data and create an accuracy column
    df = df[df["matched_out_indice"] > -1]

    # df = df[df["user_prompt_id"] == 3]

    # Parameter to see variate
    paramX = "alpha"
    paramY = "accuracy"
    paramZ = "clip_val"

    # Additional filter conditions
    # df = df[df["user_prompt_id"] == 0]

    # Take a random selection of the dataframe
    df_show = df[df["alpha"]==20][df["label"]==0][df["matched_out_indice"]==0].sample(frac=0.5, random_state=1)

    for index, row in df_show.iterrows():
        print(f"Output Text: {row['output_text']}, Matched Out Indice: {row['matched_out_indice']}")

    # Compile the accuracy from the evaluation
    if df.iloc[0]["expected_outputs"] == "AB":
        df['accuracy'] = df.apply(accuracy_from_AB_eval, axis=1)
    elif df.iloc[0]["expected_outputs"] == "open_question":
        df['accuracy'] = df.apply(accuracy_from_open_question_eval, axis=1)
    else :
        df['accuracy'] = df.apply(accuracy_from_close_question_eval, axis=1)


    # print(df[df["label"]==0][df["alpha"]==-1][df["user_prompt_id"]==0].iloc[0])
    # print(df[df["label"]==0][df["alpha"]==-1][df["user_prompt_id"]==1].iloc[0])


    # Calculate mean accuracy for each 'beta', 'clip_val', and 'label' combination
    mean_accuracy_label = df.groupby([paramX, "clip_val", "label"]).agg(
        count=(paramY, "size"),         # Count of elements in each group
        accuracy=(paramY, "mean")   # Average accuracy for each group
    ).reset_index()

    # Calculate global mean accuracy across all labels for each 'beta' and 'clip_val'
    mean_accuracy = df.groupby([paramX, "clip_val"]).agg(
        count=(paramY, "size"),
        accuracy=(paramY, "mean")
    ).reset_index()


    label_names = ["real", "fake", "all"]

    # Prepare color palette for different labels and clip values
    palette = cfg["compile"]["palette"]

    dash_styles = ["solid", "dash", "dot", "dashdot"]

    fig = go.Figure()

    # Plot accuracy for label 0, label 1, and overall mean accuracy with variations of clip_val
    for j, value in enumerate(df[paramZ].unique()):
        label_true = mean_accuracy_label[(mean_accuracy_label["label"] == 0) & (mean_accuracy_label[paramZ] == value)]
        label_false = mean_accuracy_label[(mean_accuracy_label["label"] == 1) & (mean_accuracy_label[paramZ] == value)]
        mean_accuracy_clip = mean_accuracy[mean_accuracy[paramZ] == value]

        # Plot each subset with different colors for each `clip_val`
        for i, d in enumerate([label_true, label_false, mean_accuracy_clip]):
            fig.add_trace(go.Scatter(
                x=d[paramX],
                y=d[paramY],
                mode="lines+markers",
                name=f"{label_names[i]} ({paramZ}={value})",
                hovertext=d["count"],
                line=dict(color=palette[i%len(palette)], dash=dash_styles[j%len(dash_styles)])
            ))

    # Update layout and display plot
    fig.update_layout(
        title="Mean accuracy according to steering amount, data label, and clip value",
        plot_bgcolor="lightgrey",
        xaxis_title="{}".format(paramX),
        yaxis_title="{}".format(paramY),
        legend_title="Data Label and Clip Value"
    )

    fig.show()

def accuracy_from_open_question_eval(elmt):
    """
        if matched_out_indice is 0, means the evaluation is to Yes==True==real
    """
    return not float(elmt["label"] == elmt["matched_out_indice"])
    

def accuracy_from_close_question_eval(elmt):
    """
        If label is 0, means the desired output is REAL
        If mathced_out_indice is 0, means the evaluation is to Yes
        If prompt_id is in [0, 1], then Yes == REAL
    """
    if elmt["user_prompt_id"] in [0]:
        return float(elmt["label"] == elmt["matched_out_indice"])
    else :
        return float(not elmt["label"] == elmt["matched_out_indice"])

def accuracy_from_AB_eval(elmt):
    """
        If label is 0, means the desired output is REAL
        If mathced_out_indice is 0, means the evaluation is to A 
        if the prompt is ending with "Not famous" (aka user_prompt_id in [0, 2]), Then A == REAL
    """
    if elmt["user_prompt_id"] in [0, 2]:
        return float(elmt["label"] == elmt["matched_out_indice"])
    else :
        return float(not elmt["label"] == elmt["matched_out_indice"])


def main():

    cfg = ConfigManager().config

    # Load data from the steered folder
    results_files = os.path.join(
        cfg["evaluation"]["output_folder"], 
        "steer_out_{}_evaluated.pkl".format("_".join(cfg["steering"]["evaluation_set"]))
    )

    with open(results_files, "rb") as file:
        eval = pickle.load(file)


    plot_steering_wight_and_binary_out(eval, cfg)







if __name__ == "__main__":

    main()