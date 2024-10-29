import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go

from config_manager import ConfigManager




def plot_steering_wight_and_binary_out(list_result, cfg):
    """
    Plot the steering weight and accuracy
    """


    # Convert list_result to a pandas DataFrame
    df = pd.DataFrame([elmt.__dict__ for elmt in list_result])

    # Filter out the undecisive data
    df = df[df["evaluation"]<2]
    df = df[df["clip_val"]==0]

    # Create a new column 'accuracy' that gives 0 if evaluation is equal to the label of each element
    df['accuracy'] = df.apply(lambda row: 0 if row['evaluation'] == row['label'] else 1, axis=1)

    palette = cfg["compile"]["palette"]

    print(df[df["label"] == 0].iloc[0])

    # Plot mean accuracy for the two labels 0 and 1, and mean accuracy for all the data
    mean_accuracy_label = df.groupby(["beta", "label"]).agg(
        count=("accuracy", "size"),         # Count of elements in each 'beta' group
        avg_accuracy=("accuracy", "mean")   # Average accuracy for each 'beta' group
    ).reset_index()
    label_true = mean_accuracy_label[mean_accuracy_label["label"] == 0]
    label_false = mean_accuracy_label[mean_accuracy_label["label"] == 1]

    # Global average accuracy
    mean_accuracy = df.groupby("beta").agg(
        count=("accuracy", "size"),         # Count of elements in each 'beta' group
        avg_accuracy=("accuracy", "mean")   # Average accuracy for each 'beta' group
    ).reset_index()

    label_names = ["real", "fake", "all"]

    fig = go.Figure()
    for i, d in enumerate([label_true, label_false, mean_accuracy]):
        fig.add_trace(go.Scatter(x=d["beta"], 
                                y=d["avg_accuracy"], 
                                mode="lines", 
                                name=label_names[i], 
                                hovertext=d["count"],
                                line=dict(color=palette[i])))

    fig.update_layout(
        title="Mean accuracy according to steering amount and data label",
        plot_bgcolor="grey",
        xaxis_title="Steering coeficient",
        yaxis_title="Mean accuracy",
    )

    fig.show()



    return





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