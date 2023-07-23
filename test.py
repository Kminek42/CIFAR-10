import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import learning_time_estimation as lte
import time
from model_class import ResNet, ResNetBlock

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

def prepare_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])
    
    
    test_dataset = torchvision.datasets.CIFAR10(
        root="./downloads",
        train=False,
        download=True,
        transform=transform
    )

    print(len(test_dataset))

    return test_dataset

test_dataset = prepare_dataset()
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

print(len(test_dataset))
dev = torch.device("mps")


def validate(*, model, set, dev):
    all = 0
    good = 0
    validate_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
    )
    for inputs, targets in iter(validate_loader):
        inputs, targets = inputs.to(dev), targets.to(dev)
        outputs = model.forward(inputs)
        for i in range(len(outputs)):
            if torch.argmax(outputs[i]) == targets[i]:
                good += 1
            
            else:
                img = torchvision.transforms.ToPILImage()(inputs[i])
                # plt.imshow(img)
                # plt.title(f"Real: {labels[int(targets[i])]};  NN: {labels[int(torch.argmax(outputs[i]))]}")
                # plt.show()
            all += 1

    return (good / all)