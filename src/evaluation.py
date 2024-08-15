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
    X, Y = data_to_numpy(data)

    for model in trained_models:
        # prediction results
        predictions_by_class = model.predict_proba(X)
        predictions = predictions_by_class[:, 1]
        # Compile 2nd order metrics with sklearn
        avg_prec = ["Average Precision", metrics.average_precision_score(Y, predictions)]
        roc_auc = ["Roc AUC", metrics.roc_auc_score(Y, predictions)]

        # accuracy
        acc = ["Accuracy", model.evaluate(data)]
    
        log_metrics(live, [acc, avg_prec, roc_auc] , model.model_type, split)
    
    live.log_sklearn_plot("roc", Y, predictions, name=f"roc/{split}")


def extract_hover_text(data):
    return {"hover_text": ["Data_type: {}\n Input key: {}\n Label: {}\n Sufix: {}\n Output: {}".format(
        data[i].__class__.__name__, 
        data[i].input_text, 
        data[i].label, 
        data[i].sufix,
        data[i].output_text[:min(len(data[i].output_text), 20)]) for i in range(len(data))],
            "output_length": [2*len(data[i].output_text)for i in range(len(data))]}




def generate_plots(train_data, test_data, trained_models, feature_vectors):
    
    # Extract data points and labels from training data
    X_train, Y_train = data_to_numpy(train_data)
    hover_info_train = extract_hover_text(train_data)
    predictions_train = trained_models[0].predict(X_train)

    # Extract data points and labels from test data
    X_test, Y_test = data_to_numpy(test_data)
    hover_info_test = extract_hover_text(test_data)
    predictions_test = trained_models[0].predict(X_test)

    # create a color vector that map the labels to colors
    colors = np.array(["#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#FF0000", "#800080", "#00FFFF", "#FF00FF"])
    colors_train = colors[Y_train]
    colors_test = colors[Y_test+2]

    # Create a scatter plot for training data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_train[:, 0], y=X_train[:, 1], mode='markers',
        name='Train Data',
        text=hover_info_train["hover_text"],
        hovertemplate="%{text}",
        marker=dict(color=colors_train, symbol=predictions_train, size=hover_info_train["output_length"])
    ))

    # Create a scatter plot for test data
    fig.add_trace(go.Scatter(
        x=X_test[:, 0], y=X_test[:, 1], mode='markers',
        name='Test Data',
        text=hover_info_test["hover_text"],
        hovertemplate="%{text}",
        marker=dict(color=colors_test, symbol=predictions_test, size=hover_info_test["output_length"])
    ))

    for i, vector in enumerate(feature_vectors):
        # Add the direction vector to the plot
        print("vector", vector)
        fig.add_trace(go.Scatter(
            x=[0, vector["projection_direction"][0]], y=[0, vector["projection_direction"][1]], mode='lines',
            name=vector["vector_type"] + ': Relative direction',
            line=dict(color=colors[-(2*i)], width=4)
        ))

        fig.add_trace(go.Scatter(
            x=[vector["center1"][0], vector["center2"][0]], y=[vector["center1"][1], vector["center2"][1]], mode='lines',
            name=vector["vector_type"] + ': Feature direction (center to center)',
            line=dict(color=colors[-(2*i+1)], width=1)
        ))

    # Show the plot
    fig.show()


def log_metrics(live, metric, model_name, split):
    # Log a metric using Dvc live
    for i, metric in enumerate(metric):
        print("Model {} {}: {} on {}".format(model_name, metric[0], metric[1], split))
        live.log_metric(f"Model {model_name} {metric[0]} {split}", metric[1])

        # Populate the summary
        if not live.summary:
            live.summary = {metric[0]: {}}
        elif metric[0] not in live.summary:
            live.summary[metric[0]] = {}
        if not model_name in live.summary[metric[0]]:
            live.summary[metric[0]][model_name] = {}
        live.summary[metric[0]][model_name][split] = metric[1]


def log_plots(live, projected_data):
    # Log the plot using Dvc live
    fig = go.Figure(data=go.Scatter(x=projected_data[:, 0], y=projected_data[:, 1], mode='markers'))
    hover_text = [f"Data Point {i+1}" for i in range(len(projected_data))]
    fig.update_traces(text=hover_text, hovertemplate="Info: %{text}")
    live.log_plot(fig, "projected_data")

def load_data(folder, names):
    data_list = []
    for name in names:
        # Load the data from the given file type with pickle
        with open(os.path.join(folder, name + ".pkl"), 'rb') as file:
            data_list.extend(pickle.load(file))
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
    cfgg = ConfigManager().config #["evaluation"]

    # Import the projected data and trained models from the previous stages
    train_data = load_data(cfgg["projection"]["projection_data_folder"], cfgg["experiment"]["split"]["training_data"])
    test_data = load_data(cfgg["projection"]["projection_data_folder"], cfgg["experiment"]["split"]["testing_data"])
    directions = load_data(cfgg["projection"]["projection_data_folder"], ["vectors"])
    trained_models = load_models(cfgg["detection"]["detection_model_path"])

    cfg = cfgg["evaluation"]

    # Create a DVC live instance
    live = Live(cfg["evaluation_folder"], report="html")

    generate_plots(train_data, test_data, trained_models, directions)

    # Calculate the accuracy of the trained models on all the data
    calculate_mterics(live, trained_models, test_data, split="test")
    calculate_mterics(live, trained_models, train_data, split="train")

    live.make_report()

    # Generate and log the plots and metrics


if __name__ == "__main__":
    main()