import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import learning_time_estimation as lte
import time
import model_class as mc

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
train_dataset = torchvision.datasets.CIFAR10(
    root="./downloads",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

print(len(train_dataset))

dev = torch.device("mps")

model = mc.ResNet().to(dev)
# model = torch.load(f="model.pt").to(dev)

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

torch.save(obj=model.to("cpu"), f="model.pt")
