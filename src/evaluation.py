import dvc.api
import numpy as np
import plotly.graph_objects as go
import pickle
import os
import sys

from sklearn import metrics
from dvclive import Live

from config_manager import ConfigManager
from detection import *

def data_to_numpy(some_data):
    X = np.array([data_elt.activations for data_elt in some_data])
    Y = np.array([data_elt.label for data_elt in some_data])
    return X, Y

def calculate_mterics(live, trained_models, data, split="test"):
    accuracies = {}
    X, Y = data_to_numpy(data)

    for model in trained_models:
        accuracies[model.model_type] = {}
        # Calculate accuracy for each model using its score function
        accuracies[model.model_type][split] = model.evaluate(data)

        # predictions_by_class = model.predict_proba(X)
        # predictions = predictions_by_class[:, 1]

        # # Use dvclive to log a few simple metrics...
        # avg_prec = metrics.average_precision_score(Y, predictions)
        # roc_auc = metrics.roc_auc_score(Y, predictions)

        # if not live.summary:
        #     live.summary = {"avg_prec": {}, "roc_auc": {}}
        # live.summary["avg_prec"][split] = avg_prec
        # live.summary["roc_auc"][split] = roc_auc

        # # ... and plots...
        # # ... like an roc plot...
        # live.log_sklearn_plot("roc", Y, predictions, name=f"roc/{split}")
        # # ... and precision recall plot...
        # # ... which passes `drop_intermediate=True` to the sklearn method...
        # live.log_sklearn_plot(
        #     "precision_recall",
        #     Y,
        #     predictions,
        #     name=f"prc/{split}",
        #     drop_intermediate=True,
        # )
        # # ... and confusion matrix plot
        # live.log_sklearn_plot(
        #     "confusion_matrix",
        #     Y.squeeze(),
        #     predictions_by_class.argmax(-1),
        #     name=f"cm/{split}",
        # )
    return accuracies



def generate_plots(projected_data):
    # Create a scatter plot of the projected data
    fig = go.Figure(data=go.Scatter(x=projected_data[:, 0], y=projected_data[:, 1], mode='markers'))

    # Add hover text to display info of each data point
    hover_text = [f"Data Point {i+1}" for i in range(len(projected_data))]
    fig.update_traces(text=hover_text, hovertemplate="Info: %{text}")

    # Show the plot
    fig.show()


def log_metrics(live, metric):
    # Log a metric using Dvc live
    for i, accuracy in enumerate(accuracies):
        print(f"Model {i+1} accuracy: {accuracy}")
        live.log(f"model_{i+1}_accuracy", accuracy)

def log_plots(projected_data):
    # Log the plot using Dvc live
    fig = go.Figure(data=go.Scatter(x=projected_data[:, 0], y=projected_data[:, 1], mode='markers'))
    hover_text = [f"Data Point {i+1}" for i in range(len(projected_data))]
    fig.update_traces(text=hover_text, hovertemplate="Info: %{text}")
    live.log_plot(fig, "projected_data")

def load_data(folder, name):
    # Load the data from the given file type with pickle
    with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
        data_list = pickle.load(file)
    return data_list

def load_models(folder):
    # Load the trained models from the given folder
    trained_models = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'rb') as model_file:
            model = pickle.load(model_file)
            trained_models.append(model)
    return trained_models

def main():
    cfg = ConfigManager().config

    # Import the projected data and trained models from the previous stages
    train_data = load_data(cfg["projection_data_folder"], sys.argv[1])
    test_data = load_data(cfg["projection_data_folder"], sys.argv[2])
    trained_models = load_models(cfg["detection_model_path"])

    # Create a DVC live instance
    live = Live()

    # Calculate the accuracy of the trained models on all the data
    test_metrics = calculate_mterics(live, trained_models, test_data, split="test")
    train_metrics = calculate_mterics(live, trained_models, train_data, split="train")

    print(test_metrics)
    print(train_metrics)

    # Generate and log the plots and metrics


if __name__ == "__main__":
    main()