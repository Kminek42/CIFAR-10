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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        
        self.lin1 = nn.Linear(128 * 8 * 8, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.activation(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)

        output = nn.MaxPool2d(kernel_size=(2, 2))(output)
        output = nn.Flatten()(output)
        
        output = self.lin1(output)
        output = self.activation(output)
        output = self.lin2(output)

        return output
