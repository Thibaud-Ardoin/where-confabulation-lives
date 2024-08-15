import os
import random
import sys
import yaml
import pickle

from datas import DataGenerator 
from config_manager import ConfigManager 

def main():
    # Here not sure if we need to use the param.yaml
    cfg = ConfigManager().config

    random.seed(cfg["prepare"]["seed"])

    # Initialize the dataset generator
    dg = DataGenerator()

    # Populate the data/prepared folder
    os.makedirs(cfg["prepare"]["prepared_data_folder"], exist_ok=True)
    files = cfg["experiment"]["data"]
    for input_file in files :
        input_name = os.path.basename(input_file)
        output_data = os.path.join(cfg["prepare"]["prepared_data_folder"], input_name + ".pkl")

        data = dg.load_from_type(input_file)

        # At this point maybe run the inference of the model to populate the data
        # model(data)

        with open(output_data, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        pickle_file.close()

if __name__ == "__main__":
    main()
