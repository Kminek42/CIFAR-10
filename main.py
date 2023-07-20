import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import learning_time_estimation as lte
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
    
train_dataset = torchvision.datasets.CIFAR10(
    root="./downloads",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./downloads",
    train=False,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=False
)

def validate():
    good = 0
    all = 0

    for inputs, targets in iter(test_loader):
        inputs, targets = inputs.to(dev), targets.to(dev)
        outputs = model.forward(inputs)
        for i in range(len(outputs)):
            if torch.argmax(outputs[i]) == targets[i]:
                good += 1
            all += 1
    
    return good / all


print(len(train_dataset))
print(len(test_dataset))

dev = torch.device("mps")
model = nn.Sequential(
    nn.Conv2d(3, 12, kernel_size=5, padding="same"),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),

    nn.Conv2d(12, 48, kernel_size=5, padding="same"),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(48 * 8 * 8, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

epoch_n = 10
t0 = time.time()
for epoch in range(1, epoch_n + 1):
    loss_sum = 0
    for inputs, targets in iter(train_loader):
        inputs, targets = inputs.to(dev), targets.to(dev)
        outputs = model.forward(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss

    print(f"\nEpoch: {epoch}, mean loss: {loss_sum / len(train_loader)}")
    lte.show_time(t0, epoch / epoch_n)


print(validate())
