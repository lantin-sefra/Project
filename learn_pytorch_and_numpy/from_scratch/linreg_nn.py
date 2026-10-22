"""
把 linreg.py 改成 nn.Module 版 —— 先只学 Linear，不碰 ReLU。

对照表（旧 → 新）：
  w, b 自己建              →  nn.Linear(2, 1) 里自带 weight / bias
  def linreg(X, w, b)      →  def forward(self, X)
  net(X, w, b)             →  model(X)
  sgd([w, b], ...)         →  optim.SGD(model.parameters(), ...)
  squard_loss              →  仍可手写；也可用 nn.MSELoss
"""
import torch
import torch.nn as nn
import random

torch.manual_seed(42)

# ---------- 数据和 linreg.py 完全一样 ----------
w_true = torch.tensor([2.0, -3.4])
b_true = 4.2


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


features, labels = synthetic_data(w_true, b_true, 1000)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# ---------- 只新学这一块：nn.Linear = X @ w + b ----------
# nn.Linear(in_features, out_features)
#   输入最后一维是 in_features（这里 2）
#   输出最后一维是 out_features（这里 1）
#   内部：weight 形状 (out, in)=(1,2)，bias 形状 (1,)
#   计算：y = X @ weight.T + bias   （和你写的 X @ w + b 等价）


class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 替代手写的 w、b

    def forward(self, X):
        return self.linear(X)  # 就是 Xw + b，没有别的


model = LinReg()

# 看看参数长什么样（对应以前的 w、b）
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
# linear.weight (1, 2)   ← 以前 w 是 (2,1)，这里是转置放的，别慌
# linear.bias   (1,)


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ---------- 训练：循环结构和 linreg.py 一样 ----------
lr = 0.025
num_epochs = 10
batch_size = 10

# 以前：sgd([w, b], lr, batch_size) 里会除 batch_size
# 现在：loss 用 mean()，等价于 sum() 后再 /batch_size；SGD 只负责 param -= lr * grad
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(model(X), y).mean()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    with torch.no_grad():
        train_l = squared_loss(model(features), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

# 取出学到的 w、b（weight 是 (1,2)，拉平成 (2,) 再和真值比）
w_hat = model.linear.weight.detach().reshape(-1)
b_hat = model.linear.bias.detach()
print(f"w的估计误差: {w_true - w_hat}")
print(f"b的估计误差: {b_true - b_hat}")
