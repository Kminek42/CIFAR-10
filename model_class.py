import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=(5, 5), 
                               padding="same")
        
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=(5, 5), 
                               padding="same")
        
    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x += skip
        x = nn.ReLU()(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=12, 
                               kernel_size=5, 
                               padding="same")
        
        self.block1 = ResNetBlock(12)
        self.block2 = ResNetBlock(12)
        self.block3 = ResNetBlock(12)
        self.block4 = ResNetBlock(12)
        self.block5 = ResNetBlock(12)
        self.block6 = ResNetBlock(12)
        self.block7 = ResNetBlock(12)
        self.block8 = ResNetBlock(12)
        
        self.lin1 = nn.Linear(12 * 16 * 16, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.activation(output)
        output = nn.MaxPool2d(kernel_size=(2, 2))(output)

        output = self.block1.forward(output)
        output = self.block2.forward(output)
        output = self.block3.forward(output)
        output = self.block4.forward(output)
        output = self.block5.forward(output)
        output = self.block6.forward(output)
        output = self.block7.forward(output)
        output = self.block8.forward(output)

        output = nn.Flatten()(output)
        output = self.lin1(output)
        output = self.activation(output)
        output = self.lin2(output)

        return output


