import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class ExchangeRateDataset:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.scaler = MinMaxScaler()
        self.scaled_data = None

    def scale(self):
        data = self.df['Rate'].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(data)
        return self.scaled_data

    def inverse_scale(self, data):
        return self.scaler.inverse_transform(data)

    def get_raw_data(self):
        return self.df
