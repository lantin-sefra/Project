"""
练习 5：用 nn 写完整训练循环（对应笔记 ch05_5 末尾；承接 linreg_nn.py）
任务：XOR 型数据——线性模型学不会，加一层 ReLU 就能学会。
  y = 1 if x0 * x1 > 0 else 0
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

N = 2000
X = torch.randn(N, 2)
y = ((X[:, 0] * X[:, 1]) > 0).long()
Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]


def make_model(with_relu=True):
    # TODO 1: with_relu=True  -> Linear(2,16) -> ReLU -> Linear(16,2)
    #         with_relu=False -> Linear(2,16) -> Linear(16,2)
    if (with_relu):
        return nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    else:
        return nn.Sequential(
            nn.Linear(2,16),
            nn.Linear(16,2))


def train_and_eval(model, epochs=200, lr=0.1):
    # TODO 2: 优化器吃 model.parameters()；分类 + long 标签 → CrossEntropyLoss
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        # TODO 3: 前向 → 损失 → 清梯度 → backward → step
        # loss_fn(logits, target)：先 model(Xtr) 得到 [B,2] logits，再和 ytr 算 CE
        loss = loss_fn(model(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # TODO 4:验证：返回测试集准确率（外面 assert 的就是这个数）
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1)  # 每行取 logits 最大的类
        return (pred == yte).float().mean().item()


acc_relu = train_and_eval(make_model(True))
acc_lin = train_and_eval(make_model(False))
print(f"with ReLU: {acc_relu:.3f}   without ReLU: {acc_lin:.3f}")
assert acc_relu > 0.9
assert acc_lin < 0.65

# 思考题（写在这里，用自己的话）：为什么两层 Linear 不加激活，等价于一层 Linear？
# 答：
print("ex5 OK")
