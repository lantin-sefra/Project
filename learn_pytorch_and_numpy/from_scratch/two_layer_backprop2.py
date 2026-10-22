import torch

torch.manual_seed(0)

n_in, n_hiden, n_out, p = 4, 5, 3, 8

X = torch.randn(n_in,p)
Y = torch.randn(n_out,p)

params = {
    "W1": (torch.randn(n_hiden,n_in) * 0.5).requires_grad_(True),
    "b1": (torch.randn(n_hiden) * 0.1).requires_grad_(True),
    "W2": (torch.randn(n_out,n_hiden) * 0.5).requires_grad_(True),
    "b2": (torch.randn(n_out) * 0.1).requires_grad_(True),
}

def forward() :
    Z1 = params["W1"] @ X + params["b1"].unsqueeze(1)
    H = torch.tanh(Z1)
    Z2 = params["W2"] @ H + params["b2"].unsqueeze(1)
    L = (0.5 * (Z2 - Y) ** 2).sum()
    return L

lr = 0.01
for i in range(1000):
    L = forward()
    L.backward()
    with torch.no_grad():
        for theta in (params["W1"],params["b1"],params["W2"],params["b2"]):
            theta -= lr * theta.grad
            theta.grad.zero_()
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {L.item()}")
        print(f"W1: {params['W1']}")
        print(f"b1: {params['b1']}")
        print(f"W2: {params['W2']}")
        print(f"b2: {params['b2']}")
        print("--------------------------------")
