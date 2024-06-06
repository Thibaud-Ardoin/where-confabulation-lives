from datetime import datetime
from typing import List, Optional
import os

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from config_manager import ConfigManager

@dataclass
class MyData:
    label: int = None
    user_prompt: str = None
    system_prompt: str = None
    switch_name: str = None
    id: int = None
    input_text: str = None
    activation: Optional[list] = None
    output_text: str = None


class conspiracy(MyData):
    description: str
    date : datetime = None

class celebrity(MyData):
    birthdate: datetime = None

class english_word(MyData):
    synonym: str = None




"""
    Matching each data type to a special formating to fit the desired data structure
"""
class DataGenerator():
    def __init__(self):
        self.cfg = ConfigManager().config
        self.global_id = 0


    # This section contain all the specific extraction for each text input types
    def text_type_matcher(self, text_familly, text):
        match text_familly:
            case "conspiracy":
                conspiracy_name, conspiracy_description = text.split(":")
                return conspiracy(
                    input_text = conspiracy_name,
                    description = conspiracy_description
                )
            
            case "celebrity":
                celebrity_infos = text.split(" - ")
                if len(celebrity_infos) == 1 : celebrity_infos.append(None)     # In case of no given date of birth
                return celebrity(
                    input_text = celebrity_infos[0],
                    birthdate = celebrity_infos[1]
                )
            
            case "english_word":
                word = text.split(" ")[-1]
                return english_word(
                    input_text = word
                )

            case _ :
                raise "DataGenerator -> data.py \n Looks like this type of data is not recognised"


    # Global loop over all the files that needs to be converted in datasets
    def data_loading(self):
        generated_data = []
        for type_name in self.cfg["inputs"]:
            for i, switch in enumerate(self.cfg["inputs"][type_name]["switches"]):
                file_name = os.path.join(self.cfg["text_data_folder"], "{}_{}".format(type_name, switch))
                with open(file_name) as f:
                    for lin in f:
                        str_list = list(filter(len, lin.split("\n")))
                        for text in str_list:
                            for user_prompt in self.cfg["inputs"][type_name]["user_prompts"]:
                                for system_prompt in self.cfg["inputs"][type_name]["system_prompts"]:
                                    # str
                                    data_elmt = self.text_type_matcher(type_name, text)
                                    data_elmt.switch_name = switch
                                    data_elmt.user_prompt = user_prompt
                                    data_elmt.system_prompt = system_prompt
                                    # int
                                    data_elmt.label = i
                                    data_elmt.id = self.global_id
                                    self.global_id +=1

                                    generated_data.append(data_elmt)
        return generated_data





if __name__ == "__main__": 

    cfg = ConfigManager().config

    dg = DataGenerator()
    data = dg.data_loading()

    print(data)