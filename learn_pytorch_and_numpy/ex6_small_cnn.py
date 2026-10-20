"""
练习 6：综合 —— 默写 SmallCNN（对应第 9 章主线）
先合上笔记，在纸上推每层 shape，再写。
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    [B,1,28,28] -> Conv(1->32,3,p1)+ReLU+MaxPool2 -> Conv(32->64,3,p1)+ReLU+MaxPool2
                -> Flatten -> Linear(?,128)+ReLU -> Linear(128,10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136,128),
            nn.Linear(128,10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


m = SmallCNN()
x = torch.randn(16, 1, 28, 28)
assert m.features(x).shape == (16, 64, 7, 7), "卷积部分输出 shape 不对"
assert m(x).shape == (16, 10)

# # 手算参数量：conv1 + conv2 + fc1 + fc2（weight 和 bias 都要算），先写式子再填数
# p_conv1 = ...
# p_conv2 = ...
# p_fc1 = ...
# p_fc2 = ...
n = sum(p.numel() for p in m.parameters())
# assert n == p_conv1 + p_conv2 + p_fc1 + p_fc2, f"实际总数 {n}"

mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
n_mlp = sum(p.numel() for p in mlp.parameters())
print("CNN params:", n, " MLP params:", n_mlp)
# 思考：参数量最大的是哪一层？为什么卷积层参数很少却更适合图像？
m(x).sum().backward()
assert all(p.grad is not None for p in m.parameters()), "有参数没拿到梯度——检查是否都注册了"
print("ex6 OK")
