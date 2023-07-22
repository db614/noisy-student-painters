import torch
import torch.nn as nn
import torch.nn.functional as F


# Model

class CNN(nn.Module):
    # Expects a 32 x 32 pixel image.

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4096, 2048)  # Takes the number of layers * size in pixels
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 12)  # Has to end up with a size equal to the number of classes

    def forward(self, x):
        # Perform a convolution giving us 32 output layers from the initial 3, then do a non-linear activation and pool
        # with stride of 1 to prevent reduction of output
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.norm(x)

        # Layer 2 of the CNN, this time with no normalisation after. Not clear how much difference this makes.
        x = self.pool(F.relu(self.conv2(x)))
        # We flatten to a single dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)


        return x


# TODO: Add a few more models to choose from, ResNet etc.











