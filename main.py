import xarray as xr
from dataloader import WeatherDataset
from models.model import CustomModule
from evaluation import Evaluator
import torch
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':

    time_step = 10*24
    batch_size = 1
    #predited_feature = 'z' need to make this variable

    datadir = 'C:/Users/gabri/Documents/ETH/Master/dl-project/data'
    #ds = xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords')
    ds = xr.open_mfdataset(f'{datadir}/temperature_850/*.nc', combine='by_coords')
    #ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'t': '850'}
    lead_time = 3*24  # (0 = next hour)  # 5 * 24
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

    
    '''with torch.no_grad():
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
        plt.show()'''

    # evaluate on test set

    evaluator = Evaluator(datadir, model, dg_test)
    evaluator.evaluate()


