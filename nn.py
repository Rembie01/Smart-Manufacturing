import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16385, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(nn.Tanh(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

