import math

import xarray as xr
import argparse
from dataloader import WeatherDataset
from models.lin_model import LinearModel
# from models.graph_model import GraphModel
from models.model import CustomModule
from evaluation import Evaluator
from models.graph_wavenet import gwnet
from models.wavenet import *
from models.new_graph_wavenet import *
from models.graph_model import *
from models.bfgraph_model import *
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass arguments to weather prediction framework')

    parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'val'])
    parser.add_argument('--model', default='g_wavenet', help='select model',
                        choices=['register_your_model', 'linear', 'graph', 'graph_wavenet', 'wavenet', 'g_wavenet',
                                 'bfgraph_model'])
    parser.add_argument('--data_path', default='./data', help='specify path to dataset')
    parser.add_argument('--predictions_path', default='.', help='specify where to store predictions')
    parser.add_argument('--model_path', default='./tmp', help='specify where to store models')

    parser.add_argument('--train_start_year', type=int, default=1979, help='first year used for training')
    parser.add_argument('--train_end_year', type=int, default=2016, help='last year used for training')
    parser.add_argument('--lead_time', type=int, default=3 * 24, help='time frame to predict')
    parser.add_argument('--time_step', type=int, default=12, help='number of time steps used for training')

    parser.add_argument('--training_features', default='t', help='features used for training',
                        choices=['z', 't', 'zt'])
    parser.add_argument('--predicted_feature', default='t', help='feature to predict', choices=['z', 't'])

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='train batch size')

    parser.add_argument('--load_checkpoint', default=None, help='use model checkpoint')

    args = parser.parse_args()
    print(args)
    mode = args.mode
    model_name = args.model
    train_start_year = max(args.train_start_year, 1979)
    train_end_year = min(args.train_end_year, 2016)

    time_step = args.time_step
    batch_size = args.batch_size  # 4
    # predict_feature = 'z' need to make this variable

    datadir = args.data_path  # './data'
    training_features = args.training_features
    pred_feature = args.predicted_feature
    features = []
    dic = dict()
    if 'z' in training_features:
        dic['z'] = '500'
        features.append(xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords'))
    if 't' in training_features:
        dic['t'] = '850'
        features.append(xr.open_mfdataset(f'{datadir}/temperature_850/*.nc', combine='by_coords'))
    ds = xr.merge(features, compat='override')  # Override level. discarded later anyway.

    lead_time = args.lead_time  # (0 = next hour)  # 5 * 24
    train_years = (str(train_start_year), str(train_end_year))  # ('1979', '2016')
    test_years = ('2017', '2018')

    ds_train = ds.sel(time=slice(*train_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = WeatherDataset(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size,
                              predicted_feature=pred_feature)
    dg_test = WeatherDataset(ds_test, dic, lead_time, time_steps=time_step, batch_size=batch_size,
                             mean=dg_train.mean, std=dg_train.std, shuffle=False, load=False,
                             predicted_feature=pred_feature)

    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = None

    start_epoch = 0
    num_epochs = args.epochs  # 1
    learning_rate = 1e-6
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = None

    if model_name == 'linear':
        model = LinearModel(device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # elif model_name == 'graph':
    #    model = GraphModel(device).to(device)
    #    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif model_name == 'graph_wavenet':
        model = gwnet(device, 32 * 64 * 1, dropout=0.3, supports=None, gcn_bool=True,
                      addaptadj=True, aptinit=None,
                      in_dim=1, out_dim=1, residual_channels=32, dilation_channels=32,
                      skip_channels=256,
                      end_channels=512,
                      kernel_size=2, blocks=4, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif model_name == 'custom_module':
        model = CustomModule(device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif model_name == 'wavenet':
        model = WavenetModel(input_size=(time_step, 32, 64, len(features),), num_filters= 24 * len(features),
                             kernel_size=(3, 3, 2), num_residual_blocks=int(math.log(time_step, 2)) - 1,
                             lead_time=lead_time, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif model_name == 'g_wavenet':
        model = GraphWavenetModel(input_size=(time_step, 32, 64, len(features),),
                                  num_filters=time_step * 64 * len(features),
                                  kernel_size=(1, 1, 2), num_residual_blocks=int(math.log(time_step, 2)) - 1,
                                  device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif model_name == 'bfgraph_model':
        model = BFGraph(input_size=(time_step, 32, 64, len(features)), num_filters=time_step * 64 * len(features),
                        num_residual_blocks=18, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    elif model_name == 'graph':
        model = GraphModel(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    else:
        pass

    if args.load_checkpoint:
        checkpoint = torch.load(f'{args.model_path}/{args.load_checkpoint}', map_location=device)
        print('Loaded Checkpoint:', checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    if mode == 'train':
        for epoch in range(start_epoch, num_epochs):
            for i in range(dg_train.__len__()):
                inputs, labels = dg_train.__getitem__(i)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs, labels)

                loss = criterion(outputs, labels)
                loss.backward()
                # print(model.parameters().__next__())
                optimizer.step()
                # print(model.parameters().__next__())

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item() / batch_size))
                if i % 1000 == 0:
                    with torch.no_grad():
                        evaluator = Evaluator(datadir, args.predictions_path, model, dg_test, device, pred_feature)
                        preds, true = evaluator.create_predictions(partial=True)
                        evaluator.evaluate(preds, true, plot=True,
                                           path=f'{args.model_path}/{model_name}_epoch_{epoch + 1}_batch_{i}')
                    model_save_fn = f'{args.model_path}/{model_name}_{epoch + 1}_{i}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, model_save_fn)

            model_save_fn = f'{args.model_path}/{model_name}_{epoch + 1}.pt'
            print("Saving model:", model_save_fn)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, model_save_fn)

        print('Finished Training')
        with torch.no_grad():
            evaluator = Evaluator(datadir, args.predictions_path, model, dg_test, device, pred_feature)
            preds, true = evaluator.create_predictions(partial=True)
            evaluator.evaluate(preds, true, plot=True)

    if mode == 'val':
        if not args.load_checkpoint:
            raise AttributeError("No checkpoint specified for testing")
        with torch.no_grad():
            evaluator = Evaluator(datadir, args.predictions_path, model, dg_test, device, pred_feature)
            preds, true = evaluator.create_predictions()
            evaluator.evaluate(preds, true)
