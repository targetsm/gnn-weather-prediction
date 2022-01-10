import torch
from evaluation import Evaluator
import math
from dataloader_chris import WeatherDataset
from models.new_graph_wavenet import *
import xarray as xr


MODEL_PATH = 'models'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'

train_start_year = 1979
train_end_year = 2016

batch_size=4
time_step = 12

datadir = './data' # './data'
training_features = 't'
pred_feature = 't'
features = []
dic = dict()
if 'z' in training_features:
    dic['z'] = '500'
    features.append(xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords'))
if 't' in training_features:
    dic['t'] = '850'
    features.append(xr.open_mfdataset(f'{datadir}/temperature_850/temperature_850_5.625deg/*.nc', combine='by_coords'))
ds = xr.merge(features, compat='override')  # Override level. discarded later anyway.

lead_time = 3 * 24  # (0 = next hour)  # 5 * 24
train_years = (str(train_start_year), str(train_end_year))  # ('1979', '2016')
test_years = ('2017', '2018')

ds_train = ds.sel(time=slice(*train_years))
ds_test = ds.sel(time=slice(*test_years))

dg_train = WeatherDataset(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size,
                          predicted_feature=pred_feature)
dg_test = WeatherDataset(ds_test, dic, lead_time, time_steps=time_step, batch_size=batch_size,
                          mean=dg_train.mean, std=dg_train.std, shuffle=False, load=False,
                          predicted_feature=pred_feature)


# load model
model = GraphWavenetModel(input_size=(time_step, 32, 64, len(features),),
                            num_filters=time_step * 64 * len(features),
                            kernel_size=(1, 1, 2), num_residual_blocks=int(math.log(time_step, 2)) - 1,
                            device=device).to(device)


model.load_state_dict(torch.load(f'{MODEL_PATH}/g_wavenet_4_20000.pt'), strict=False)
model.to(device)


with torch.no_grad():
    evaluator = Evaluator(datadir, '.', model, dg_test, device, pred_feature)
    preds, true = evaluator.create_predictions(partial=False)
    evaluator.evaluate(preds, true, plot=True)