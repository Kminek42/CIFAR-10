import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels_in, 
                               out_channels=channels_in, 
                               kernel_size=(kernel, kernel), 
                               padding="same")
        
        self.conv2 = nn.Conv2d(in_channels=channels_in, 
                               out_channels=channels_out, 
                               kernel_size=(kernel, kernel), 
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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")

        self.block1 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        self.block2 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        self.block3 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        self.block4 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        self.block5 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        self.block6 = ResNetBlock(channels_in=128, channels_out=128, kernel=3)
        
        self.lin1 = nn.Linear(128 * 8 * 8, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.activation(output)
        
        output = nn.MaxPool2d(kernel_size=(2, 2))(output)

        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)

        output = nn.MaxPool2d(kernel_size=(2, 2))(output)
        output = nn.Flatten()(output)
        
        output = self.lin1(output)
        output = self.activation(output)
        output = self.lin2(output)

        return output
