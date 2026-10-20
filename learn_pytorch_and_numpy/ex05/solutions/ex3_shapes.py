import torch
import torch.nn as nn

x = torch.randn(8, 3, 32, 32)

# 1) Conv2d(3, 16, k=5, s=2, p=0)
a = (8, 16, 14, 14)
assert nn.Conv2d(3, 16, 5, stride=2)(x).shape == a

# 2) Conv2d(3, 16, k=3, s=1, p=1) 再 MaxPool2d(2)
b = (8, 16, 16, 16)
assert nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.MaxPool2d(2))(x).shape == b

# 3) 两个 Conv(k=3,p=1) + 两个 MaxPool(2)，使输出为 [8,64,8,8]
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
)
assert net(x).shape == (8, 64, 8, 8)

# 4) Flatten 之后的特征数
flat = 64 * 8 * 8
assert nn.Flatten()(net(x)).shape == (8, flat)

# 5) AdaptiveAvgPool2d((1,1)) 后接 Flatten
c = (8, 64)
assert nn.Sequential(net, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())(x).shape == c
print("ex3 OK")
