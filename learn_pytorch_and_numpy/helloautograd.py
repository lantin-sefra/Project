import torch

x = torch.tensor([2.,3.],requires_grad=True)
print(x.requires_grad,x.is_leaf)

x = torch.tensor(2.,requires_grad=True)
y = x**2 + 3*x + 1
print(y)

y.backward()
print(y.item(),x.grad)

#向量先约成标量再求导
x = torch.tensor([1.,2.,3.],requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)

x = torch.tensor(2.0,requires_grad=True)
with torch.no_grad():
    y = x**2
print(y.requires_grad)

y = x**2
y.backward()
print(x.grad)

print("--------------------------------")
x = torch.tensor(3.0,requires_grad=True)
y = x**2
out = y.detach() *x

out.backward()
print(x.grad)

print("--------------------------------")
x = torch.tensor([1.,1.],requires_grad = True)
y = (x[0] + 2*x[1]) ** 2
y.backward()
print(x.grad)


