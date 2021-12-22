import xarray as xr
from dataloader import WeatherDataset
from models.model import CustomModule
import torch
import numpy as np
import matplotlib.pyplot as plt


def create_iterative_predictions(model, dg, max_lead_time=5 * 24):
    """Create iterative predictions"""
    state = dg.data[:dg.n_samples]
    preds = []
    for _ in range(max_lead_time // dg.lead_time):
        state = model.predict(state)
        p = state * dg.std.values + dg.mean.values
        preds.append(p)
    preds = np.array(preds)

    lead_time = np.arange(dg.lead_time, max_lead_time + dg.lead_time, dg.lead_time)
    das = []
    lev_idx = 0
    for var, levels in dg.var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, :, lev_idx],
                dims=['lead_time', 'time', 'lat', 'lon'],
                coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, :, :, :, lev_idx:lev_idx + nlevs],
                dims=['lead_time', 'time', 'lat', 'lon', 'level'],
                coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon,
                        'level': levels},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das)

def create_predictions(model, dg):
    """Create non-iterative predictions"""
    x_test, y_test = next(iter(dg))
    preds = model(x_test, y_test)[0]
    # Unnormalize
    preds = preds * dg.std.values + dg.mean.values
    das = []
    lev_idx = 0
    for var, levels in dg.var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx],
                dims=['time', 'lat', 'lon'],
                coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            print(das)
            print(preds.shape)
            print(levels)
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx:lev_idx+nlevs],
                dims=['time', 'lat', 'lon', 'level'],
                coords={'time': dg.valid_time[:1], 'lat': dg.ds.lat, 'lon': dg.ds.lon, 'level': [levels]},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    print(da_true)
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    print('value:', error.to_array().values)
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse

def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Load the test dataset. If z return z500, if t return t850.
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if var in ['z', 't']:
        try:
            ds = ds.drop('level')
        except ValueError:
            ds = ds.drop('level')
    return ds.sel(time=years)

if __name__ == '__main__':

    time_step = 3*24
    batch_size = 4
    #predited_feature = 'z' need to make this variable

    datadir = 'C:/Users/gabri/Documents/ETH/Master/dl-project/data'
    #ds = xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords')
    ds = xr.open_mfdataset(f'{datadir}/temperature_850/*.nc', combine='by_coords')
    #ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'t': '850'}
    lead_time = 5*24  # (0 = next hour)  # 5 * 24
    # train_years = ('1979', '2015')
    train_years = ('2013', '2015')
    test_years = ('2017', '2018')

    ds_train = ds.sel(time=slice(*train_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = WeatherDataset(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size)
    dg_test = WeatherDataset(ds_test, dic, lead_time, time_steps=time_step, batch_size=batch_size, mean=dg_train.mean,
                            std=dg_train.std, shuffle=False)
    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    model = CustomModule()

    num_epochs = 1
    learning_rate = 1e-6
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, data in enumerate(dg_train, 0):
            inputs, labels = data
            labels = labels[:,:,:,:,[0]]

            optimizer.zero_grad()

            outputs = model(inputs, labels)
            #plt.imshow(outputs.detach().numpy()[-1, -1, :, :, 0])
            #plt.show()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

            if i == 100:
                break

    print('Finished Training')

    with torch.no_grad():
        model.eval()
        x_test, y_test = next(iter(dg_test))
        y_test = y_test[:,:,:,:,[0]]
        prediction = model(x_test, y_test)
        print(y_test.shape)
        #prediction = x_test[:,-2:-1,:,:,[0]]
        print(prediction.shape, x_test.shape)
        error = np.sqrt(torch.sum((y_test - prediction) ** 2).numpy() / y_test.numpy().size)

        print(error)

        plt.imshow(prediction[-1, -1])
        plt.show()
        plt.imshow(y_test[-1,-1])
        plt.show()

        # evaluate on test set

    pred_save_fn = f'{datadir}/predictions'
    # Create predictions
    pred = create_predictions(model, dg_test)
    print(f'Saving predictions: {pred_save_fn}')
    pred.to_netcdf(pred_save_fn)
    print(pred.to_array(), )
    # Print score in real units
    # TODO: Make flexible for other states
    valid = load_test_data(f'{datadir}/temperature_850', 't').isel(time=lead_time+time_step)
    #t850_valid = load_test_data(f'{datadir}temperature_850', 't')
    #valid = xr.merge([z500_valid, t850_valid], compat='override')
    print(valid)
    print(compute_weighted_rmse(pred, valid).load().to_array().values)


