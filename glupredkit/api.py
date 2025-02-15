import requests
import numpy as np
import pandas as pd
from io import BytesIO
import glupredkit.helpers.cli as helpers


# DATA PARSING AND RETRIEVAL
def get_synthetic_data():
    url = 'https://raw.githubusercontent.com/miriamkw/GluPredKit/main/example_data/synthetic_data.csv'
    response = requests.get(url).content
    return pd.read_csv(BytesIO(response), index_col="date", parse_dates=True, low_memory=False)


def get_parsed_data(file_name):
    data = helpers.read_data_from_csv("data/raw/", file_name)
    return data



