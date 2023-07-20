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
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        self.conv2 = nn.Conv2d(in_channels=12, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        self.conv3 = nn.Conv2d(in_channels=12, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        self.conv4 = nn.Conv2d(in_channels=12, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        self.conv5 = nn.Conv2d(in_channels=12, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        
        self.lin1 = nn.Linear(12 * 16 * 16, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.activation(output)
        output = nn.MaxPool2d(kernel_size=(2, 2))(output)

        skip = output
        
        
        output = self.conv2(output)
        output = self.activation(output)

        output = self.conv3(output)
        output += skip
        skip = output
        output = self.activation(output)
        
        
        output = self.conv4(output)
        output = self.activation(output)

        output = self.conv5(output)
        output += skip
        skip = output
        output = self.activation(output)


        output = nn.Flatten()(output)
        output = self.lin1(output)
        output = self.activation(output)
        output = self.lin2(output)

        return output


dev = torch.device("mps")
model = torch.load(f="model.pt").to(dev)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
good = 0
all = 0

for inputs, targets in iter(test_loader):
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

print(good / all)
