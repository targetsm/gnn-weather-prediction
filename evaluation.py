import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

class Evaluator:
    """
    Evaluate test set for given model and compute scores
    """

    def __init__(self, datadir, model, dg_test):
        self.datadir = datadir
        self.model = model
        self.dg_test = dg_test
        model.eval()

    def evaluate(self):
        pred_save_fn = f'{self.datadir}/predictions'
        # Create predictions
        pred = self.create_predictions(self.model, self.dg_test)
        print(f'Saving predictions: {pred_save_fn}')
        pred.to_netcdf(pred_save_fn)
        print(pred.to_array(), )
        # Print score in real units
        # TODO: Make flexible for other states
        valid = self.load_test_data(f'{self.datadir}/temperature_850', 't').isel(time=self.dg_test.lead_time + self.dg_test.time_steps)
        # t850_valid = load_test_data(f'{datadir}temperature_850', 't')
        # valid = xr.merge([z500_valid, t850_valid], compat='override')
        print(valid)
        print(self.compute_weighted_rmse(pred, valid).load().to_array().values)

    def create_iterative_predictions(self, model, dg, max_lead_time=5 * 24):
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

    def create_predictions(self, model, dg):
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

    def compute_weighted_rmse(self, da_fc, da_true, mean_dims=xr.ALL_DIMS):
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

    def load_test_data(self, path, var, years=slice('2017', '2018')):
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

    def print_sample(self):
        x_test, y_test = next(iter(self.dg_test))
        y_test = y_test[:, :, :, :, [0]]
        prediction = self.model(x_test, y_test)
        print(y_test.shape)
        # prediction = x_test[:,-2:-1,:,:,[0]]
        print(prediction.shape, x_test.shape)
        error = np.sqrt(torch.sum((y_test - prediction) ** 2).numpy() / y_test.numpy().size)

        print(error)

        plt.imshow(prediction[-1, -1])
        plt.show()
        plt.imshow(y_test[-1, -1])
        plt.show()