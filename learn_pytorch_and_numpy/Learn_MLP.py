import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim = 784, hidden = 128, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP()
print(model)
x = torch.randn(32,784)

