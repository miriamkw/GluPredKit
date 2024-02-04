from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import pandas as pd


def get_data_frame_for_type(data, type, name):
    df = data[data.type == type]
    df = df.copy()

    # Converting to mg/dL if necessary
    if name == "CGM":
        try:
            unit = df.iloc[0]['unit']
            if str(unit).startswith('mmol'):
                df['value'] = df['value'].apply(lambda x: x * 18.018)
        except IndexError as ie:
            print(f"Error: There are no blood glucose values in the dataset. Make sure to specify the correct start "
                  f"date in the parser. {ie}")
            raise
        except KeyError as ke:
            print(f"Error: The key '{ke}' does not exist in the DataFrame.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    df.rename(columns={'value': name}, inplace=True)
    df.rename(columns={'startDate': 'date'}, inplace=True)
    df.drop(['type', 'sourceName', 'sourceVersion', 'unit', 'creationDate', 'device', 'endDate'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    return df


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        data = pd.DataFrame(record_list)


        return df
