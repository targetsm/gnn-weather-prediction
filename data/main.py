# import necessary packages
from dataloader import WeatherDataset
from model import CustomModule
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


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
        inputs, labels = data

        inputs = (inputs - torch.from_numpy(train_dataset.mean.values) )/ torch.from_numpy(train_dataset.std.values) # should be axis specific, right now not correct...
        labels = (labels - torch.from_numpy(train_dataset.mean.values) )/ torch.from_numpy(train_dataset.std.values) # should be axis specific, right now not correct...

        optimizer.zero_grad()

        outputs = model(inputs, labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')


model.eval()
x_test, y_test = next(iter(testloader))
prediction = model(x_test, y_test)

error = np.sqrt(torch.sum((y_test - prediction)**2).numpy()/y_test.numpy().size)


print(error)

plt.imshow(prediction[-1, -1])
plt.show()

# evaluate on test set

