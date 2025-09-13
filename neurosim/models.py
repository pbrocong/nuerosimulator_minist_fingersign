import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """MNIST: 784 → 128 → 10"""
    def __init__(self, weight_min=-1.0, weight_max=1.0):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        for layer in [self.fc1, self.fc2]:
            nn.init.uniform_(layer.weight, a=weight_min, b=weight_max)
            if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.tanh(self.bn1(self.fc1(x)))
        return torch.log_softmax(self.fc2(x), dim=1)

class SimpleCNN(nn.Module):
    """Sign-MNIST: 간단 CNN"""
    def __init__(self, weight_min=-1.0, weight_max=1.0, num_classes=24):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*14*14, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(m.weight, a=weight_min, b=weight_max)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim()==3: x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = x.view(-1, 32*14*14)
        x = torch.relu(self.bn2(self.fc1(x)))
        return self.fc2(x)