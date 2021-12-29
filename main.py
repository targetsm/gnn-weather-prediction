import xarray as xr
import argparse
from dataloader import WeatherDataset
from models.lin_model import CustomModule
from evaluation import Evaluator
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass arguments to weather prediction framework')

    parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'val'])
    parser.add_argument('--model', default='linear', help='select model',
                        choices=['register_your_model', 'linear', 'graph_wavenet'])
    parser.add_argument('--data_path', default='./data', help='specify path to dataset')
    parser.add_argument('--predictions_path', default='.', help='specify where to store predictions')

    parser.add_argument('--train_start_year', type=int, default=2013, help='first year used for training')
    parser.add_argument('--train_end_year', type=int, default=2016, help='last year used for training')

    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='train batch size')

    args = parser.parse_args()
    print(args)
    mode = args.mode
    model_name = args.model
    train_start_year = max(args.train_start_year, 1979)
    train_end_year = min(args.train_end_year, 2016)

    time_step = 5 * 24
    batch_size = args.batch_size  # 4
    # predict_feature = 'z' need to make this variable

    datadir = args.data_path  # './data'
    # z = xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords')
    ds = xr.open_mfdataset(f'{datadir}/temperature_850/*.nc', combine='by_coords')
    # ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    dic = {'t': '850'}  # dic = {'z':'500', 't': '850'}
    lead_time = 3 * 24  # (0 = next hour)  # 5 * 24
    # train_years = ('1979', '2015')
    train_years = (str(train_start_year), str(train_end_year))  # ('2013', '2016')
    test_years = ('2017', '2018')

    ds_train = ds.sel(time=slice(*train_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = WeatherDataset(ds_train, dic, lead_time, time_steps=time_step, batch_size=batch_size)
    dg_test = WeatherDataset(ds_test, dic, lead_time, time_steps=time_step, batch_size=ds_test.__len__(),
                             mean=dg_train.mean, std=dg_train.std, shuffle=False, load=False)

    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = None

    num_epochs = args.epochs  # 1
    learning_rate = 1e-6
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = None

    if model_name == 'linear':
        model = CustomModule(device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if mode == 'test':
            # load your model here in case of only testing
            pass
    elif model_name == 'graph_wavenet':
        pass
    else:
        pass

    if mode == 'train':
        for epoch in range(num_epochs):
            for i in range(dg_train.__len__()):
                inputs, labels = dg_train.__getitem__(i)
                inputs = inputs.to(device)
                labels = labels[:, :, :, :, [0]].to(device)

                optimizer.zero_grad()

                outputs = model(inputs, labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item() / batch_size))

        print('Finished Training')

    # evaluate on test set
    preddir = args.predictions_path
    evaluator = Evaluator(datadir, preddir, model, dg_test, device)
    evaluator.evaluate()
    evaluator.print_sample()
