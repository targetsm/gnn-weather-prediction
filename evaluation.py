import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

class Evaluator:
    """Evaluate test set for given model and compute scores"""

    def __init__(self, datadir, preddir, model, dg_test, device):
        self.datadir = datadir
        self.preddir = preddir
        self.model = model
        self.dg_test = dg_test
        self.device = device
        model.eval()

    def evaluate(self):
        """Perform evaluation on the test set"""
        pred_save_fn = f'{self.preddir}/predictions'

        # Create predictions
        pred = self.create_predictions(self.model, self.dg_test)
        print(f'Saving predictions: {pred_save_fn}')
        pred.to_netcdf(pred_save_fn)

        # Print score in real units
        valid = self.load_test_data(f'{self.datadir}/temperature_850', 't')
        print('RMSE:',self.compute_weighted_rmse(pred, valid).load().to_array().values)

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
                    coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat,
                            'lon': dg.ds.lon},
                    name=var
                ))
                lev_idx += 1
            else:
                nlevs = len(levels)
                das.append(xr.DataArray(
                    preds[:, :, :, :, lev_idx:lev_idx + nlevs],
                    dims=['lead_time', 'time', 'lat', 'lon', 'level'],
                    coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat,
                            'lon': dg.ds.lon,
                            'level': levels},
                    name=var
                ))
                lev_idx += nlevs
        return xr.merge(das)

    def create_predictions(self, model, dg):
        """Create non-iterative predictions"""
        x_test, y_test = next(iter(dg))
        x_test = x_test.to(self.device)
        y_test = y_test[:,:,:,:,[0]].to(self.device)
        preds = model(x_test, y_test)[0].cpu()
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
                das.append(xr.DataArray(
                    preds[:, :, :, lev_idx:lev_idx+nlevs],
                    dims=['time', 'lat', 'lon', 'level'],
                    coords={'time': dg.valid_time[:1], 'lat': dg.ds.lat, 'lon': dg.ds.lon,
                            'level': [levels]},
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
        error = da_fc - da_true
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
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
            ds = ds.drop('level')
        return ds.sel(time=years)

    def print_sample(self):
        """Plot sample comparison"""
        x_test, y_test = next(iter(self.dg_test))
        y_test = y_test[:, :, :, :, [0]].to(self.device)
        x_test = x_test.to(self.device)
        prediction = self.model(x_test, y_test).cpu()

        plt.imshow(prediction[-1, -1])
        plt.show()
        plt.imshow(y_test[-1, -1].cpu())
        plt.show()
