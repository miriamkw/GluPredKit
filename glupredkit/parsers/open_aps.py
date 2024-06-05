"""
The Open APS parser is processing the .pkl file produced by using the data cleaner by Harry Emerson: https://github.com/hemerson1/OpenAPS_Cleaner.
"""
import pandas as pd
import pickle
from .base_parser import BaseParser


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the processed data including <file_name>.pkl.
        """
        print("FILE PATH", file_path)

        with open(file_path, 'rb') as f:
            loaded_object = pickle.load(f)

        df = loaded_object

        print(df)

        return df
