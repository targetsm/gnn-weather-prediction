import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    def __init__(self, ds, var_dict, lead_time, time_steps=128, batch_size=32, shuffle=True,
                 load=True, mean=None, std=None, predicted_feature='z'):

        self.ds = ds
        self.var_dict = var_dict
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        labels = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        i = 0
        for var, levels in var_dict.items():
            if var == predicted_feature:
                self.label = i
            data.append(ds[var].expand_dims({'level': generic_level}, 1))
            i += 1

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, - self.time_steps - self.lead_time - 1)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(self.time_steps + lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.n_samples - self.time_steps - self.lead_time) / (self.batch_size)))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size: (i + 1) * self.batch_size]
        x_li = [self.data.isel(time=idxs + j).values for j in range(self.time_steps)]
        # X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.time_steps + self.lead_time - 1).values[:,:,:,[self.label]]
        X = torch.from_numpy(np.stack(x_li, axis=1))
        y = torch.from_numpy(np.expand_dims(y, axis=1))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
