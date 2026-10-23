import torch


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


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d, q = 4, 3, 2
    X = torch.randn(n, d)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    W = torch.randn(d, q)
    b = torch.zeros(q)

    y_hat = softreg(X, W, b)
    loss = cross_entropy(y_hat, y)
    dW, db = grad_cross_entropy(X, y_hat, y)
    optimizer([W, b], [dW, db], lr=0.5)

    print("loss:", float(loss))
    print("y_hat:\n", y_hat)
    print("dW:\n", dW)
    print("db:", db)
