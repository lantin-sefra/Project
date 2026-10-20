import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_classes):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


m = DeepMLP(784, [128, 64], 10)
out = m(torch.randn(4, 784))
assert out.shape == (4, 10)
n = sum(p.numel() for p in m.parameters())
assert n == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10, n
assert len(list(m.parameters())) == 6
print("ex1 OK, params =", n)
