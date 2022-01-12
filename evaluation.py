import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

class Evaluator:
    """Evaluate test set for given model and compute scores"""

    def __init__(self, datadir, preddir, model, dg_test, device, predicted_feature):
        self.datadir = datadir
        self.preddir = preddir
        self.model = model
        self.dg_test = dg_test
        self.device = device
        self.pred_feature = predicted_feature
        dg_test.data.load()
        model.eval()

    def evaluate(self, pred, valid, plot=False, path='tmp.png'):
        """Perform evaluation on the test set"""
        # Print score in real units
        print('RMSE:',self.compute_weighted_rmse(pred, valid).load().to_array().values)
        if plot:
            self.print_sample(path)

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

    def create_predictions(self, partial=False):
        """Perform evaluation on the test set"""
        pred_save_fn = f'{self.preddir}/predictions'
        true_save_fn = f'{self.preddir}/truth'

        # Create predictions

        pred = []
        true = []
        for i in range(self.dg_test.__len__()):
            inputs, labels = self.dg_test.__getitem__(i)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs, labels)
            pred.append(outputs)
            true.append(labels)
            print(i, self.dg_test.__len__())
            if partial and i == 10:
                break
        pred = torch.cat(pred)
        true = torch.cat(true)

        preds = pred.to('cpu') * self.dg_test.std.values + self.dg_test.mean.values
        das = []
        lev_idx = 0
        preds = preds.squeeze(axis=1)
        for var, levels in self.dg_test.var_dict.items():
            if var == self.pred_feature:
                das.append(xr.DataArray(
                    preds[:, :, :, [lev_idx]],
                    dims=['time', 'lat', 'lon', 'level'],
                    coords={'time': self.dg_test.valid_time[:preds.size(0)], 'lat': self.dg_test.ds.lat,
                            'lon': self.dg_test.ds.lon,
                            'level': [levels]},
                    name=var
                ))
            lev_idx += 1
        pred = xr.merge(das)

        preds = true.to('cpu') * self.dg_test.std.values + self.dg_test.mean.values
        das = []
        lev_idx = 0
        preds = preds.squeeze(axis=1)
        for var, levels in self.dg_test.var_dict.items():
            if var == self.pred_feature:
                print(preds.shape)
                das.append(xr.DataArray(
                    preds[:, :, :, [lev_idx]],
                    dims=['time', 'lat', 'lon', 'level'],
                    coords={'time': self.dg_test.valid_time[:preds.size(0)], 'lat': self.dg_test.ds.lat,
                            'lon': self.dg_test.ds.lon,
                            'level': [levels]},
                    name=var
                ))
            lev_idx += 1
        true = xr.merge(das)

        print(f'Saving predictions: {pred_save_fn}')
        pred.to_netcdf(pred_save_fn)
        print(f'Saving true values: {true_save_fn}')
        true.to_netcdf(true_save_fn)

        return pred, true

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
        print(error)
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
        rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
        return rmse

    def print_sample(self, path):
        """Plot sample comparison"""
        x_test, y_test = next(iter(self.dg_test))
        y_test = y_test[:, :, :, :, [0]].to(self.device)
        x_test = x_test.to(self.device)
        prediction = self.model(x_test, y_test).cpu()

        plt.imshow(prediction[-1, -1])
        plt.savefig(f'{path}_prediction.png', dpi=100)
        plt.show()

        plt.imshow(y_test[-1, -1].cpu())
        plt.savefig(f'{path}_true.png', dpi=100)
        plt.show()


class Graph_Model_Evaluator:
    """Evaluate test set for given model and compute scores"""

    def __init__(self, datadir, preddir, model, dg_test, edge_index, device):
        self.datadir = datadir
        self.preddir = preddir
        self.model = model
        self.dg_test = dg_test
        self.device = device
        self.edge_index = edge_index.to(self.device)
        model.eval()

    def evaluate(self):
        """Perform evaluation on the test set"""
        pred_save_fn = f'{self.preddir}/predictions'

        # Create predictions
        pred = self.create_predictions(self.model, self.dg_test)
        print(f'Saving predictions: {pred_save_fn}')
        pred.to_netcdf(pred_save_fn)

        # Print score in real units
        valid = self.load_test_data(f'{self.datadir}/temperature_850/temperature_850_5.625deg', 't')
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
        preds = model(x_test, self.edge_index)[0].cpu()
        # Unnormalize
        preds = preds.detach() * dg.std.values + dg.mean.values
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
        prediction = self.model(x_test, self.edge_index.to(self.device)).detach().cpu()
        print(prediction.shape)
        plt.imshow(prediction[-1, -1][:, :, 0])
        plt.show()
        plt.imshow(y_test[-1, -1][:, :, 0].cpu())
        plt.show()
