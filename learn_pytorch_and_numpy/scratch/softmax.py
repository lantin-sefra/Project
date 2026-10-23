import torch
import torch.nn as nn

def softmax(O):
    O_exp = torch.exp(O)
    partition = O_exp.sum(dim=1, keepdim=True)
    return O_exp / partition


def softreg(X, W, b):
    """Softmax 回归：O = XW + b，再按行 softmax。"""
    return softmax(X @ W + b)


def cross_entropy(y_hat, y):
    """y 为 one-hot；对 batch 取平均（经验风险）。"""
    return -(y * torch.log(y_hat)).sum(dim=1).mean()


def grad_cross_entropy(X, y_hat, y):
    dO = y_hat - y
    dW = X.T @ dO
    db = dO.sum(dim=0)
    return dW, db


def optimizer(params, grads, lr):
    """SGD：param ← param - lr * grad（梯度里已含 1/n，不再除 batch_size）。"""
    with torch.no_grad():
        for param, grad in zip(params, grads):
            param -= lr * grad



class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, X):
        return self.fc(X)

    def loss(self, logits, y):
        return self.criterion(logits, y)

