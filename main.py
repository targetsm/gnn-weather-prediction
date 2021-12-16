import xarray as xr
import matplotlib.pyplot as plt
import torch
from torch.nn import Transformer
import numpy as np


# load data
z500 = xr.open_mfdataset('data/geopotential_500/*.nc', combine='by_coords')
z500.z.isel(time=0).plot()
plt.show()

# Generate train and test data

prev_time = 100
lead_time = 3
train_samples = 2

train_dataset = z500.sel(time=slice('2015', '2015'))
data_mean = train_dataset.isel(time=slice(0, None, 10000)).mean().load()
data_std = train_dataset.isel(time=slice(0, None, 10000)).std().load()

train_dataset = (train_dataset - data_mean) / data_std

train_data = train_dataset.to_array().to_numpy()
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2]*train_data.shape[3]).transpose(1,0,2)

test_dataset = z500.sel(time=slice('2016', '2017'))

test_dataset = (test_dataset - data_mean) /data_std
test_data = test_dataset.to_array().to_numpy()
originalt_test_shape = test_data.shape
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2]*test_data.shape[3]).transpose(1,0,2)

n = prev_time+lead_time
train_data_list = [train_data[i:i+n] for i in range(train_data.shape[0]-n)]
train_data_batch = np.array(train_data_list[-train_samples:]).squeeze().transpose(1,0,2)
print(train_data_batch.shape)

'''
X_train = torch.from_numpy(train_data[-1200-120: -120])
y_train = torch.from_numpy(train_data[-120:])

X_test = torch.from_numpy(test_data[-1200-120: -120])
y_test = torch.from_numpy(test_data[ -120:])
'''

X_train = torch.from_numpy(train_data_batch[: -lead_time])
y_train = torch.from_numpy(train_data_batch[-lead_time:])

X_test = torch.from_numpy(test_data[-prev_time-lead_time: -lead_time])
y_test = torch.from_numpy(test_data[ -lead_time-1:])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# training

transformer_model = Transformer(d_model=2048)
transformer_model.train()
for i in range(X_train.shape[1]):
    print(i)
    out = transformer_model(X_train[:, [i]], y_train[:,[i]]) # need to change here, by selecting i we lose one dimension i thingk

# Generate test predictions

src_mask = Transformer.generate_square_subsequent_mask(lead_time+1)
transformer_model.eval()
output = torch.from_numpy(np.full(y_test.shape, X_test[-1])[:lead_time+1])
print(output.shape)
with torch.no_grad():
    for i in range(1,lead_time+1):
        print(i)
        output[i] = transformer_model(X_test, output[:i], tgt_mask=src_mask[:i,:i])[i-1]
        #print(output)

# Evaluation

print(output.shape)
print((output - y_test[:lead_time+1])**2)

import sklearn
from sklearn.metrics import mean_squared_error
errors = []

for i in range(lead_time+1):
    print(y_test[i].shape, data_std.to_array().values)
    ys = y_test[i].squeeze()*data_std.to_array().values + data_mean.to_array().values # prob not correct...
    preds = output[i].squeeze() * data_std.to_array().values + data_mean.to_array().values
    print(ys, preds)
    errors.append(np.sqrt(sklearn.metrics.mean_squared_error(ys,  preds)))

#print(errors[73], errors[120])
plt.plot(errors[1:])

plt.show()
# compute rmse
# add mean and variance again, reshape data and plot data
# plot error