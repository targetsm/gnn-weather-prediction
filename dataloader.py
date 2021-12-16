class WeatherDataset(Dataset):
    def __init__(self, test=False, transform, lead_time):
        if test:
            self.dataset = xr.open_mfdataset('geopotential_500/*.nc', combine='by_coords').sel(time=slice('2016', '2017'))
        else:
            self.dataset = xr.open_mfdataset('geopotential_500/*.nc', combine='by_coords').sel(time=slice('1990', '2015'))
        self.transform = transform
        self.lead_time = lead_time


    def __len__(self):
        return self.dataset.sizes['time']-21

    def __getitem__(self, idx):
        if self.transform:
            sample = ToTensor()(self.dataset.to_array()[:,i: i+20].to_numpy().squeeze()).squeeze()
            label = ToTensor()(self.dataset.to_array()[:,i+20].to_numpy().squeeze()).squeeze()
        else:
            sample = torch.from_numpy(self.dataset.to_array()[:,i: i+20].to_numpy().squeeze())
            label = torch.from_numpy(self.dataset.to_array()[:,i+20].to_numpy().squeeze())
        return sample, label