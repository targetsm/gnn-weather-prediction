import xarray as xr
from dataloader import WeatherDataset
# from models.model import CustomModule
from models.GAN import CustomModule, GAT
from evaluation import Evaluator
import torch
from geometry_GNN import build_cube_edges
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    device = 'cpu'

    time_step = 10
    batch_size = 4
    save_and_load_edges = False
    #predict_feature = 'z' need to make this variable

    datadir = './data'
    #z = xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords')
    ds = xr.open_mfdataset(f'{datadir}/temperature_850/temperature_850_5.625deg/*.nc', combine='by_coords')
    #ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'t': '850'} # dic = {'z':'500', 't': '850'}
    lead_time = 3*24  # (0 = next hour)  # 5 * 24, since we want to predict 3/5 days
    # train_years = ('1979', '2015')
    train_years = ('2013', '2016')
    test_years = ('2017', '2018')

    ds_train = ds.sel(time=slice(*train_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = WeatherDataset(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size)
    dg_test = WeatherDataset(ds_test, dic, lead_time, time_steps=time_step, batch_size=batch_size,
                             mean=dg_train.mean, std=dg_train.std, shuffle=False)
    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    if save_and_load_edges: 
        # load and save or load edge indices (same for each graph). 
        # Delete the file if something relevant to edge creation changed.
        if os.path.isfile(datadir + '/edge_indices.pt'):
            edge_index = torch.load(datadir + '/edge_indices.pt')
        else:
            edge_index = build_cube_edges(
                width=64, height=32, depth=time_step, mode="8_neighbors", connect_time_steps="nearest"
            )
            edge_index_extended = torch.tensor(edge_index).repeat(4, 1).T.contiguous()
            edge_index_addon = torch.repeat_interleave(torch.tensor([i*(time_step*32*64) for i in range(batch_size)]), len(edge_index)).repeat(2, 1)
            edge_index = edge_index_extended + edge_index_addon

            torch.save(edge_index, datadir + '/edge_indices.pt')
    else:
        edge_index = build_cube_edges(
                width=64, height=32, depth=time_step, mode="8_neighbors", connect_time_steps="nearest"
            )

    edge_index_extended = torch.tensor(edge_index).repeat(4, 1).T.contiguous()
    edge_index_addon = torch.repeat_interleave(torch.tensor([i*(time_step*32*64) for i in range(batch_size)]), len(edge_index)).repeat(2, 1)
    edge_index = edge_index_extended + edge_index_addon

    model = GAT(num_features=1, num_vertices=32*64*time_step, batch_size=batch_size)
    # model = CustomModule()

    num_epochs = 5
    learning_rate = 1e-6
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dg_train):
            inputs, labels = data
            labels = labels[:,:,:,:,[0]]

            optimizer.zero_grad()
            # outputs = model(inputs.to(device), labels.to(device))
            outputs = model(inputs.to(device), edge_index.to(device))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i%5 == 0: # every 100th, print
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, loss.item()/batch_size))


    print('Finished Training')

    # evaluate on test set
    evaluator = Evaluator(datadir, model, dg_test)
    evaluator.evaluate()
    evaluator.print_sample()


