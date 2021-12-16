import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    def __init__(self, data_path, sample_time = 20, lead_time = 2, test=False):
        if test:
            self.dataset = xr.open_mfdataset(data_path + 'geopotential_500/*.nc', combine='by_coords').sel(time=slice('2016', '2017'))
        else:
            self.dataset = xr.open_mfdataset(data_path + '/geopotential_500/*.nc', combine='by_coords').sel(time=slice('1990', '2015'))
        self.dataarray = self.dataset.to_array()
        self.sample_time = sample_time
        self.mean = self.dataarray.mean()
        self.std = self.dataarray.std()
        self.lead_time = lead_time

    def __len__(self):
        return self.dataset.sizes['time']-self.lead_time-self.sample_time

    def __getitem__(self, idx):
        sample = torch.from_numpy(np.moveaxis(self.dataarray[:,idx: idx+self.sample_time].values, 0, -1))
        label = torch.from_numpy(np.moveaxis(self.dataarray[:,idx+self.sample_time:idx+self.sample_time+self.lead_time].values, 0, -1))
        return sample, label


if __name__ == '__main__':
    training_data = WeatherDataset(data_path="data/")
    print(training_data.mean, training_data.std)
    print(training_data.dataarray)

    sample, label = next(iter(training_data))
    print(sample.shape, label.shape)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    samples, labels = next(iter(train_dataloader))
    print(samples.shape, labels.shape)
