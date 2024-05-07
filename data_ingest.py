import pandas as pd

class IngestData:
    def __init__(self):
        self.data_path = None
    def get_data(self, data_path, encoding='utf-8'):
        self.data_path = data_path
        df = pd.read_csv(self.data_path, encoding = encoding)
        return df
