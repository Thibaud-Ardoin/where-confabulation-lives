import yaml
import os
import numpy as np

class ConfigManager:
    _instance = None
    _file_root = ""

    def __new__(cls, experiment_name='params'):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(experiment_name)
        return cls._instance

    def _load_config(self, experiment_name):
        config_path = os.path.join(self._file_root, f'{experiment_name}.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.pretty_format()

    def pretty_format(self):
        # Make shortcut for selecting all layers
        if "inference" in self.config and "layer" in self.config["inference"] :
            if self.config["inference"]["layers"] == "all":
                self.config["inference"]["layers"] = list(range(32))

        # Give a random run number to each work
        self.config["run_id"] = int(np.random.rand()*1000)

    def get(self, key, default=None):
        return self.config.get(key, default)


if __name__ == "__main__":

    config = ConfigManager().config
    print(config)