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

test_dataset = torchvision.datasets.CIFAR10(
    root="./downloads",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

print(len(test_dataset))

dev = torch.device("mps")
model = torch.load(f="model.pt").to(dev)

good = 0
all = 0

for inputs, targets in iter(test_loader):
    inputs, targets = inputs.to(dev), targets.to(dev)
    outputs = model.forward(inputs)
    for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == targets[i]:
            good += 1
        all += 1

print(good / all)
