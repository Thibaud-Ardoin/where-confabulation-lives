import os
import random
import sys
import yaml
import pickle

from datas import DataGenerator 
from config_manager import ConfigManager 

def main():
    # Here not sure if we need to use the param.yaml
    # params = yaml.safe_load(open("params.yaml"))["prepare"]
    cfg = ConfigManager().config

    # Take as input name of types of text data. Ex: "english_word", "celebrity"
    if len(sys.argv) < 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    random.seed(cfg["seed"])

    # Initialize the dataset generator
    dg = DataGenerator()

    # Populate the data/prepared folder
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
    for input_file in sys.argv[1:] :
        input_name = os.path.basename(input_file)
        output_data = os.path.join("data", "prepared", input_name + ".pkl")

        data = dg.load_from_type(input_file)

        # At this point maybe run the inference of the model to populate the data
        # model(data)

        with open(output_data, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        pickle_file.close()

if __name__ == "__main__":
    main()
