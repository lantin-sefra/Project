import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


m = SmallCNN()
x = torch.randn(16, 1, 28, 28)
assert m.features(x).shape == (16, 64, 7, 7)
assert m(x).shape == (16, 10)
n = sum(p.numel() for p in m.parameters())
assert n == 320 + 18496 + 401536 + 1290, n

mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
n_mlp = sum(p.numel() for p in mlp.parameters())
print("CNN params:", n, " MLP params:", n_mlp)
assert n_mlp == 101770
m(x).sum().backward()
assert all(p.grad is not None for p in m.parameters())
print("ex6 OK")
