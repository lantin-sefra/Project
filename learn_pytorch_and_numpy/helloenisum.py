import torch

# A = torch.randn(2,3)
# B = torch.randn(3,4)

# print(torch.einsum("ik,kj->ij", A, B).shape)
# print((A @ B).shape == torch.einsum("ik,kj->ij", A, B).shape,)

# print(torch.einsum("ij->ji", A).shape)
# print(A.T)

# print(torch.einsum("ij->", A))
# print(A.sum())

x = torch.randn(2,4,5)

time_mean = torch.einsum("BTC->BC",x).divide(x.shape[1])
print(time_mean.shape)
print(time_mean)
print(x.mean(dim=1))


x = torch.randn(32,20)
print(f"x: {x.shape}")
w = torch.randn(10,20)
print(f"w: {w.shape}")
b = torch.randn(10)
print(f"b: {b.shape}")
y = x @ w.T + b
print(f"y: {y.shape}")

print(torch.nn.functional.linear(x, w, b).shape == y.shape)


