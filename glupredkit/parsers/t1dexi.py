"""
The T1DEXI parser is processing the .xpt data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import pandas as pd
import os
import datetime


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the OhioT1DM dataset root folder.
        subject_id -- the id of the subject.
        year -- the version year for the dataset.
        """

        return

    def resample_data(self, tree, is_test):
        return