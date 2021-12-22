import xarray as xr
from dataloader import WeatherDataset
from models.model import CustomModule
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    time_step = 10*24

    batch_size = 4

    datadir = 'D:/datasets/WeatherBench'
    z = xr.open_mfdataset("C:/Users/gabri/Documents/ETH/Master/dl-project/data/geopotential_500/*.nc", combine='by_coords')
    #t = xr.open_mfdataset(f'{datadir}/temperature_850/temperature_850_5.625deg/*.nc', combine='by_coords')
    #ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'z': '500'}
    lead_time = 5*24  # (0 = next hour)  # 5 * 24
    # train_years = ('1979', '2015')
    train_years = ('2013', '2015')
    test_years = ('2016', '2018')

    ds_train = z.sel(time=slice(*train_years))
    ds_test = z.sel(time=slice(*test_years))

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

            optimizer.zero_grad()

            outputs = model(inputs, labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

            if i == 0:
                break

    print('Finished Training')

    model.eval()
    x_test, y_test = next(iter(dg_test))
    prediction = model(x_test, y_test)

    error = np.sqrt(torch.sum((y_test - prediction) ** 2).numpy() / y_test.numpy().size)

    print(error)

    plt.imshow(prediction[-1, -1])
    plt.show()
    plt.imshow(y_test[-1][-1])
    plt.show()

    # evaluate on test set




