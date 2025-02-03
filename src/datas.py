from datetime import datetime
from typing import List, Optional
import os
from dateutil import parser

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from config_manager import ConfigManager
import yaml
import pickle

@dataclass
class MyData:
    label: int = None
    original_name: str = None
    user_prompt: str = None
    user_prompt_id : int = None
    system_prompt: str = None
    system_prompt_id : int = None
    switch_name: str = None
    id: int = None
    input_text: str = None
    activation: Optional[list] = None
    output_text: str = None
    sufix: str = None
    input_token_length: int = None
    description: str = None
    
    def get_dialog(self):
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": self.get_user_prompt()}, 
        ]

    def get_user_prompt(self):
        return self.user_prompt.format(self.input_text)
    
    def update(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)


class truthfulQA(MyData):
    correct: List[str] = None
    wrong: List[str] = None

class hallucination_caa(MyData):
    correct: List[str] = None
    wrong: List[str] = None

class conspiracy(MyData):
    description: str
    date : datetime = None

class celebrity(MyData):
    birthdate: datetime = None

class emotion(MyData):
    emotion: str = None

class sentiment(MyData):
    emotion: str = None

class english_word(MyData):
    synonym: str = None

class french_word(MyData):
    synonym: str = None

class date(MyData):
    date : datetime = None 

class election(MyData):
    winner: str = None
    date: datetime = None
    country: str = None

class character(MyData):
    description: str = None

class medical(MyData):
    description: str = None

class legal(MyData):
    description: str = None

class protein(MyData):
    description: str = None

class culture(MyData):
    description: str = None

class poem(MyData):
    description: str = None

class airport(MyData):
    description: str = None


"""
    Matching each data type to a special formating to fit the desired data structure
"""
class DataGenerator():
    def __init__(self):
        self.cfg = ConfigManager().config["prepare"]
        with open(self.cfg["prompt_file"], 'r') as file:
            self.prompts = yaml.safe_load(file)
        self.global_id = 0


    # This section contain all the specific extraction for each text input types
    def text_type_matcher(self, text_familly, text):
        match text_familly:
            case "truthfulQA":
                data_dict = pickle.loads(eval(text))
                elmt = truthfulQA(
                    input_text = data_dict.get("Question")
                )
                elmt.correct = data_dict.get("Correct Answers")
                elmt.wrong = data_dict.get("Incorrect Answers")
                return elmt

            case "hallucination_caa":
                data_dict = pickle.loads(eval(text))
                elmt = hallucination_caa(
                    input_text = data_dict.get("question")
                )
                elmt.correct = data_dict.get("answer_not_matching_behavior")
                elmt.wrong = data_dict.get("answer_matching_behavior")
                return elmt


            case "conspiracy":
                conspiracy_name, conspiracy_description = text.split(":")
                return conspiracy(
                    input_text = conspiracy_name,
                    description = conspiracy_description
                )
            
            case "medical":
                medical_info = text.split(" | ")
                return medical(
                    input_text = medical_info[0],
                    description = medical_info[1],
                )
            
            case "culture":
                culture_info = text.split(" | ")
                return culture(
                    input_text = culture_info[0],
                    description = culture_info[1],
                )
            
            case "poem":
                return poem(
                    input_text = text,
                )
                        
            case "airport":
                return airport(
                    input_text = text,
                )
            
            case "legal":
                legal_info = text.split(" | ")
                return legal(
                    input_text = legal_info[0],
                    description = legal_info[1],
                )
            
            
            case "protein":
                return protein(
                    input_text = text
                )
            
            
            case "celebrity":
                celebrity_infos = text.split(" - ")
                if len(celebrity_infos) == 1 : celebrity_infos.append(None)     # In case of no given date of birth
                return celebrity(
                    input_text = celebrity_infos[0],
                    birthdate = celebrity_infos[1]
                )
            
            case "test_celebrity":
                celebrity_infos = text.split(" - ")
                if len(celebrity_infos) == 1 : celebrity_infos.append(None)     # In case of no given date of birth
                return celebrity(
                    input_text = celebrity_infos[0],
                    birthdate = celebrity_infos[1]
                )
            
            case "test_celebrity2":
                return celebrity(
                    input_text = text,
                )

            case "just_test":
                return celebrity(
                    input_text = text,
                )
            
            case "test_sentiment":
                return sentiment(
                    input_text = text,
                )

            case "emotion":
                return emotion(
                    input_text = text,
                )


            case "english_word":
                word = text.split(" ")[-1]
                return english_word(
                    input_text = word
                )
            
            case "french_word":
                word = text
                return french_word(
                    input_text = word
                )

            case "date":
                d = text.split(" ")[-1]
                return date(
                    input_text = d
                )
            
            case "election":
                infos = text.split(",")
                assert len(infos)==3
                parsed_date = parser.parse(infos[1])
                return election(
                    country = infos[0],
                    input_text = "{} in {}".format(infos[0], parsed_date.year),
                    date = parsed_date,
                    winner = infos[2],
                )

            case "character":
                infos = text.split(" - ")
                return character(
                    description = infos[-1],
                    input_text = infos[0],
                )

            case _ :
                raise "DataGenerator -> data.py \n Looks like this type of data is not recognised"



    # Global loop over all the files that needs to be converted in datasets
    def data_loading(self):
        generated_data = []
        for type_name in self.prompts:
            generated_data.extend(self.aggregate_type_data(type_name))
        return generated_data
    
    def aggregate_data_list(self, data_list):
        generated_data = []
        for type_name in data_list:
            generated_data.extend(self.aggregate_type_data(type_name))
        return generated_data

    def aggregate_type_data(self, type_name):
        generated_data = []
        for i, switch in enumerate(self.prompts[type_name]["switches"]):
                file_name = os.path.join(self.cfg["text_data_folder"], "{}_{}".format(type_name, switch))
                with open(file_name) as f:
                    for lin in f:
                        str_list = list(filter(len, lin.split("\n")))
                        for text in str_list:
                            for up_id, user_prompt in enumerate(self.prompts[type_name]["user_prompts"]):
                                for sp_id, system_prompt in enumerate(self.prompts[type_name]["system_prompts"]):
                                    # str
                                    data_elmt = self.text_type_matcher(type_name, text)
                                    data_elmt.original_name = type_name
                                    data_elmt.switch_name = switch
                                    data_elmt.user_prompt = user_prompt
                                    data_elmt.user_prompt_id = up_id
                                    data_elmt.system_prompt = system_prompt
                                    data_elmt.system_prompt_id = sp_id
                                    data_elmt.expected_outputs = self.prompts[type_name]["expected_outputs"]
                                    if "sufix" in self.prompts[type_name]:
                                        data_elmt.sufix = self.prompts[type_name]["sufix"]
                                    # int
                                    data_elmt.label = i
                                    data_elmt.id = self.global_id
                                    self.global_id +=1

                                    generated_data.append(data_elmt)
        return generated_data

    def load_from_type(self, file_name):
        return self.aggregate_type_data(file_name)


if __name__ == "__main__": 

    # cfg = ConfigManager().config
    cfg = ConfigManager(experiment_name="params")
    cfg._load_config(experiment_name="params")
    cfg = cfg.config

    dg = DataGenerator()
    data = dg.aggregate_type_data("medical")

    print(data[0])