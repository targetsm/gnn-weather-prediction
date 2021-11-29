import torch
import xarray as xr
import os


class WeatherDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels, load_one_year=True):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y


def touch_all_data2(save_directory, feature_dict, resulution_string, year_train_li, year_test_li, constants_loc):
    data_li = []
    for feature in feature_dict:
        for year in year_train_li + year_test_li:
            if 'path_add' in feature_dict[feature]:
                data_dir = os.path.join(save_directory, feature, feature + resulution_string,
                                        feature_dict[feature]['path_add'] + '_' + str(year) + resulution_string + '.nc')
            else:
                data_dir = os.path.join(save_directory, feature, feature + resulution_string,
                                        feature + '_' + str(year) + resulution_string + '.nc')

            print(data_dir)
            data = xr.open_mfdataset(data_dir)
            mean = data.mean(('time', 'lat', 'lon')).compute()
            std = data.std('time').mean(('lat', 'lon')).compute()
            # Normalize
            data = (data - mean) / std
            print(data)
            print("mean:", mean)
            print("std:", std)
            break

    data = xr.open_mfdataset(os.path.join(save_directory, constants_loc))




def touch_all_data(save_directory, feature_dict, resulution_string, year_train_li, year_test_li, constants_loc):
    dataset = []
    for feature in feature_dict:

        data_dir = os.path.join(save_directory, feature, feature + resulution_string, '*.nc')
        print(data_dir)
        feature_data = xr.open_mfdataset(data_dir, combine='by_coords')
        print(feature_data)
        dataset.append(feature_data)

    ds = xr.merge(dataset)
    #ds_train = ds.sel(time=slice(*year_train_li))
    ds_test = ds.sel(time=slice(*year_test_li))

    #print(ds_train)
    print(ds_test)

    data = xr.concat(ds_test, 'level').transpose('time', 'lat', 'lon', 'level')
    mean = data.mean(('time', 'lat', 'lon')).compute()
    std = data.std('time').mean(('lat', 'lon')).compute()
    # Normalize
    data = (data - mean) / std
    print(mean)
    print(std)
    print(data)




def main():
    """params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    data = torch.load('D:/datasets/WeatherBench/2m_temperature/2m_temperature_5.625deg/2m_temperature_1979_5.625deg.nc')
    training_generator = torch.utils.data.DataLoader(data, **params)
    for local_batch in training_generator:
        print(local_batch)"""

    save_directory = 'D:/datasets/WeatherBench'
    constants_loc = 'constants/constants_5.625deg.nc'
    resulution_string = '_5.625deg'
    year_train = [str(i) for i in range(1979, 2017)]
    year_test = [str(i) for i in range(2018, 2019)]

    feature_dict = {
        '2m_temperature': {'mean': 0, 'var': 0},
        '10m_u_component_of_wind': {'mean': 0, 'var': 0},
        '10m_v_component_of_wind': {'mean': 0, 'var': 0},
        'geopotential': {'mean': 0, 'var': 0},
        'geopotential_500': {'mean': 0, 'var': 0, 'path_add': 'geopotential_500hPa'},
        'potential_vorticity': {'mean': 0, 'var': 0},
        'relative_humidity': {'mean': 0, 'var': 0},
        'specific_humidity': {'mean': 0, 'var': 0},
        'temperature': {'mean': 0, 'var': 0},
        'temperature_850': {'mean': 0, 'var': 0, 'path_add': 'temperature_850hPa'},
        'toa_incident_solar_radiation': {'mean': 0, 'var': 0},
        'total_cloud_cover': {'mean': 0, 'var': 0},
        'total_precipitation': {'mean': 0, 'var': 0},
        'u_component_of_wind': {'mean': 0, 'var': 0},
        'v_component_of_wind': {'mean': 0, 'var': 0},
        'vorticity': {'mean': 0, 'var': 0},
    }

    touch_all_data2(save_directory, feature_dict, resulution_string, year_train, year_test, constants_loc)


if __name__ == '__main__':
    main()
