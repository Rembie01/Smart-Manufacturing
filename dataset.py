import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class TurbineDataset(Dataset):
    def __init__(self, data_dir='Data', turbine_number=1, transform=None):
        files = [None, 'Turbine1.csv', 'Turbine2.csv', 'Turbine3.csv', 'Turbine4.csv', 'Turbine5.csv', 'Turbine6.csv']
        self.dataframe = pd.read_csv(os.path.join(data_dir, files[turbine_number]), header=None, float_precision='high')
        print(self.dataframe.head())
        self.dataframe.columns = ['year', 'speed'] + [f'Sensor_{i-2}' for i in range(2, len(self.dataframe.columns))]
        self.transform = transform

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        time = self.dataframe.iloc[idx, 0]
        speed = self.dataframe.iloc[idx, 1]
        sensor_readings = torch.FloatTensor(self.dataframe.iloc[idx, 2:])
        if self.transform:
            sensor_readings = self.transform(sensor_readings)
        return time, speed, sensor_readings
