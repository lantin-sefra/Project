import torch
import torch.nn as nn
import torch.nn.functional as F


net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),    # 1
    nn.ReLU(),                         # 2
    nn.MaxPool2d(2),                   # 3
    nn.Conv2d(32, 64, 3, padding=1),   # 4
    nn.ReLU(),                         # 5
    nn.MaxPool2d(2),                   # 6
)


x = torch.randn(8, 3, 32, 32)
for i in range(1, len(net) + 1):
    print(i, net[:i](x).shape)
