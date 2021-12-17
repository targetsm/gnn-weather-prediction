# import necessary packages
from dataloader import WeatherDataset
from model import CustomModule
import torch
from torch.utils.data import DataLoader
import numpy as np


# Load data

train_dataset = WeatherDataset(data_path="C:/Users/gabri/Documents/ETH/Master/dl-project/data/")
test_dataset = WeatherDataset(data_path="C:/Users/gabri/Documents/ETH/Master/dl-project/data/", test=True)

train_batch_size = 64
test_batch_size = 32
trainloader = DataLoader(train_dataset, train_batch_size, shuffle=True)
testloader = DataLoader(test_dataset, test_batch_size, shuffle=False) # Frage ist wirklich, ob sie es so gemacht haben...


#(load model) maybe later

model = CustomModule()

num_epochs = 1
learning_rate = 1e-6
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = (inputs - torch.from_numpy(train_dataset.mean.values) )/ torch.from_numpy(train_dataset.std.values) # should be axis specific, right now not correct...
        labels = (labels - torch.from_numpy(train_dataset.mean.values) )/ torch.from_numpy(train_dataset.std.values) # should be axis specific, right now not correct...

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, labels)
        print(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

        if i == 2:
            break

print('Finished Training')

model.eval()
x_test, y_test = next(iter(testloader))
prediction = model(x_test, y_test)


import sklearn
errors = []

error = np.sqrt(sklearn.metrics.mean_squared_error(y_test,  prediction))

# evaluate on test set

