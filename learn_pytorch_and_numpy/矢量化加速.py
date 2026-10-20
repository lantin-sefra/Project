import math
import torch
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    # N(mu, sigma^2) 的密度：常数 * exp(- (x-mu)^2 / (2 sigma^2))
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * torch.exp(-0.5 / sigma**2 * (x - mu) ** 2)


x = torch.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]

plt.figure(figsize=(4.5, 2.5))
for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x.numpy(), y.numpy(), label=f"mean {mu}, std {sigma}")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend()
plt.show()
