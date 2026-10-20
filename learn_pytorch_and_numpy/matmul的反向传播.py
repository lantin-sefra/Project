import torch

m,n,p = 3,4,5
A = torch.randn(m,n,requires_grad=True)
X = torch.randn(n,p,requires_grad=True)
B = torch.randn(p,p,requires_grad=True)
Y = torch.randn(m,p) # 真实值,不需要梯度

E = A @ X @ B - Y
L = 1/2.0 *(E ** 2).sum()
L.backward()
print(torch.allclose(X.grad, A.T @ (E.detach() @ B.T)))
print(torch.allclose(B.grad,  X.T @ A.T @ E.detach()))


M = torch.randn(m,n)
X = torch.randn(n,p,requires_grad=True)
b = torch.randn((p,1),requires_grad=True)
o = torch.randn((m,1)).fill_(1.)
print(o,o.shape)
print((o @ b.T).shape)
Y = torch.randn(m,p)
E = M @ X + (o @ b.T) - Y
L = 1/2.0 *(E ** 2).sum()
L.backward()
print(torch.allclose(b.grad, E.T @ o))
print(torch.allclose(X.grad, M.T @E))