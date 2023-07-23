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
    torchvision.transforms.ToTensor()
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

stats = torch.zeros((3, 2))
print(stats)
input()
l = len(train_dataset)
i = 0
for img, label in train_dataset:
    for color, channel in enumerate(img):
        stats[color][0] += torch.mean(channel) / l
        stats[color][1] += torch.std(channel) / l

    i += 1
    print(i)

print(stats)

