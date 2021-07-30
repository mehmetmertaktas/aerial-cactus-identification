from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=(3, 3))
        self.fc = nn.Linear(
            in_features = 20*6*6,
            out_features = 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20*6*6)
        x = F.log_softmax(self.fc(x), dim=1)
        return x
