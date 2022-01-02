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

    def evaluate(self, pred, valid, plot=False):
        """Perform evaluation on the test set"""
        # Print score in real units
        print('RMSE:',self.compute_weighted_rmse(pred, valid).load().to_array().values)
        if plot:
            self.print_sample()

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
