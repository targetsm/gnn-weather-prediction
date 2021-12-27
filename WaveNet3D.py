import os
import time
import xarray as xr
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Activation, Add, Conv1D, Conv3D, Dense, Flatten, Input, Multiply, Concatenate
from tensorflow.keras.models import Model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, time_steps=128, batch_size=32, shuffle=True, load=True, mean=None,
                 std=None):

        self.ds = ds
        self.var_dict = var_dict
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        for var, levels in var_dict.items():
            data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, - self.time_steps - self.lead_time - 1)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        print(self.data.isel(time=slice(0, -lead_time)).shape[0])
        print(int(np.floor((self.n_samples - self.time_steps - self.lead_time) / (self.batch_size))))
        print(self.n_samples, self.time_steps, self.lead_time, self.batch_size)

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
        y = self.data.isel(time=idxs + self.time_steps + self.lead_time - 1).values
        X = tf.stack(x_li, axis=1)
        y = tf.expand_dims(y, axis=1)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


def WaveNetResLayer(num_filters, kernel_size, dilation_rate):
    def build_residual_block(l_input):
        # Gated activation.
        l_sigmoid_conv3d = Conv3D(num_filters, kernel_size, dilation_rate=dilation_rate, padding="same",
                                  activation="sigmoid")(l_input)
        l_tanh_conv3d = Conv3D(num_filters, kernel_size, dilation_rate=dilation_rate, padding="same",
                               activation="tanh")(l_input)
        l_mul = Multiply()([l_sigmoid_conv3d, l_tanh_conv3d])
        # Branches out to skip unit and residual output.
        l_skip_connection = Conv3D(num_filters, 1)(l_mul)
        l_residual = Add()([l_input, l_skip_connection])
        return l_residual, l_skip_connection

    return build_residual_block


def build_wavenet_model(input_size, num_filters, kernel_size, num_residual_blocks):
    l_input = Input(shape=input_size)

    l_stack_conv3d = Conv3D(num_filters, kernel_size, padding="same")(l_input)
    l_skip_connections = []
    for i in range(num_residual_blocks):
        l_stack_conv3d, l_skip_connection = WaveNetResLayer(
            num_filters, kernel_size, 2 ** (i + 1))(l_stack_conv3d)
        l_skip_connections.append(l_skip_connection)
    l_sum = Add()(l_skip_connections)
    relu = Activation("relu")(l_sum)
    l1_conv3d = Conv3D(input_size[-1], 1, activation="relu")(relu)
    l2_conv3d = Conv3D(input_size[-1], 1)(l1_conv3d)

    l3_conv3d = Conv3D(input_size[-1], (input_size[0], 1, 1))(l2_conv3d)

    l_output = l3_conv3d

    model = Model(inputs=l_input, outputs=l_output)
    print(model.summary(line_length=120))
    return model


def load_data(batch_size):
    return None, None


def train(wave_net_model, data, epochs):
    wave_net_model.compile(tf.keras.optimizers.Adam(), 'mse')
    history = wave_net_model.fit(data[0], epochs=epochs, validation_data=data[1])
    wave_net_model.save("./model")
    return history


def main():
    # feature_size = 8 * 13 + 6 * 1 + 5 * 1
    # build_model((64, 32, 64, feature_size,), feature_size, 2, 5)
    # build_model((128, 32, 64, feature_size,), feature_size, 2, 6)
    feature_size = 2 * 1
    time_step = 5 * 24
    temporal_model = build_wavenet_model((time_step, 32, 64, feature_size,), 2*feature_size, (2, 3, 3), 6)

    batch_size = 4

    datadir = './data'
    z = xr.open_mfdataset(f'{datadir}/geopotential_500/geopotential_500_5.625deg/*.nc', combine='by_coords')
    t = xr.open_mfdataset(f'{datadir}/temperature_850/temperature_850_5.625deg/*.nc', combine='by_coords')
    ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'z': '500', 't': '500'}
    lead_time = 1  # (0 = next hour)  # 5 * 24
    # train_years = ('1979', '2015')
    train_years = ('2013', '2015')
    test_years = ('2016', '2018')

    ds_train = ds.sel(time=slice(*train_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = DataGenerator(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size)
    dg_test = DataGenerator(ds_test, dic, lead_time, time_steps=time_step, batch_size=batch_size, mean=dg_train.mean,
                            std=dg_train.std, shuffle=False)
    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')
    train(temporal_model, (dg_train, dg_test), 1)


if __name__ == '__main__':
    gpu = True
    start_time = time.time()

    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main()
    end_time = time.time()

    # convert from seconds into hours, minutes, seconds
    print(
        f'Runtime: {int((end_time - start_time) // 3600)}h '
        f'{int(((end_time - start_time) % 3600) // 60)}m '
        f'{int((end_time - start_time) % 60)}s.'
    )
