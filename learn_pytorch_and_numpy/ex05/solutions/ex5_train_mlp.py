import torch
import torch.nn as nn

torch.manual_seed(0)

N = 2000
X = torch.randn(N, 2)
y = ((X[:, 0] * X[:, 1]) > 0).long()
Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]


def make_model(with_relu=True):
    layers = [nn.Linear(2, 16)]
    if with_relu:
        layers.append(nn.ReLU())
    layers.append(nn.Linear(16, 2))
    return nn.Sequential(*layers)


def train_and_eval(model, epochs=200, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        loss = loss_fn(model(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(dim=1) == yte).float().mean().item()


acc_relu = train_and_eval(make_model(True))
acc_lin = train_and_eval(make_model(False))
print(f"with ReLU: {acc_relu:.3f}   without ReLU: {acc_lin:.3f}")
assert acc_relu > 0.9
assert acc_lin < 0.65
print("ex5 OK")
