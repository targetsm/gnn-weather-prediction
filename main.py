import xarray as xr
import os
import numpy as np
from dataloader_old import WeatherDataset
from models.GAN import GAT
from evaluation import Graph_Model_Evaluator
import torch
from geometry_GNN import build_cube_edges
import argparse



if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    time_step = 12
    batch_size = 4
    save_and_load_edges = False
    #predict_feature = 'z' need to make this variable
    parser = argparse.ArgumentParser(description='Pass arguments to weather prediction framework')
    parser.add_argument('--data_path', default='./data', help='specify path to dataset')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')
    args = parser.parse_args()
    datadir = args.data_path
    device = args.device

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

    edge_index_extended = torch.tensor(edge_index).repeat(batch_size, 1).T.contiguous()
    edge_index_addon = torch.repeat_interleave(torch.tensor([i*(time_step*32*64) for i in range(batch_size)]), len(edge_index)).repeat(2, 1)
    print(edge_index_extended.shape, edge_index_addon.shape)
    edge_index = edge_index_extended + edge_index_addon

    model = GAT(num_features=1, num_vertices=32*64*time_step, batch_size=batch_size)
    # model = CustomModule()

    num_epochs = 10
    learning_rate = 1e-5
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.5, weight_decay=0.0001853649179441988)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    model.to(device)

    log_size = 1000

    for epoch in range(num_epochs):
        losses = []
        for i in range(len(dg_train)):
            inputs, labels = dg_train[i]
            labels = labels[:,:,:,:,[0]].to(device)

            optimizer.zero_grad()
            # outputs = model(inputs.to(device), labels.to(device))
            outputs = model(inputs.to(device), edge_index.to(device))

            loss = criterion(outputs, labels)
            losses.append(loss.item()/batch_size)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            if i%log_size == 0:      
                print('[%d, %5d] average training loss: %.3f' %
                    (epoch + 1, i + 1, np.mean(losses)))
                losses = []

    print('Finished Training')

    # evaluate on test set
    preddir ='.'
    evaluator = Graph_Model_Evaluator(datadir, preddir, model, dg_test, edge_index, device)
    evaluator.evaluate()
    evaluator.print_sample()
