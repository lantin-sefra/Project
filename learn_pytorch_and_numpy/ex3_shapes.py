"""
练习 3：shape 推导（对应笔记 ch05_4）
规则：每一题先把预测写进变量（只用笔算，不许先跑），再运行 assert 验证。
公式：H_out = floor((H_in + 2p - k) / s) + 1
"""
import torch
import torch.nn as nn

x = torch.randn(8, 3, 32, 32)

# TODO 1) Conv2d(3, 16, kernel_size=5, stride=2, padding=0)
a = torch.Size([8,16,14,14])
assert nn.Conv2d(3, 16, 5, stride=2)(x).shape == a

# TODO 2) Conv2d(3, 16, 3, padding=1) 然后 MaxPool2d(2)
b = torch.Size([8,16,16,16])
assert nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.MaxPool2d(2))(x).shape == b

# TODO 3) 用 Conv2d(k=3,p=1) 和 MaxPool2d(2) 搭一个 net，使 net(x) 的 shape 是 (8, 64, 8, 8)
net = nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.MaxPool2d(2),nn.MaxPool2d(2))
assert net(x).shape == (8, 64, 8, 8)

# TODO 4) net 输出接 nn.Flatten() 后，特征数是多少？（用于下一层 Linear 的 in_features）
flat = 4096
assert nn.Flatten()(net(x)).shape == (8, flat)

# TODO 5) 在 net 后接 AdaptiveAvgPool2d((1,1)) 再 Flatten，输出 shape？
c = torch.Size([8,64])
assert nn.Sequential(net, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())(x).shape == c
print("ex3 OK")
