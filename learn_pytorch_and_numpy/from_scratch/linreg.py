import torch
import random
from matplotlib import pyplot as plt


"""这里是真实值"""
w_true = torch.tensor([2,-3.4])
b_true = 4.2

"""数据生成"""
def synthetic_data(w,b,num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b #加法广播
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

"""生成数据集"""
features, labels = synthetic_data(w_true,b_true,1000)
print(features.shape)
print(labels.shape)

"""小批量读取数据"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

"""初始化参数"""
w = torch.normal(0,0.02,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad = True)

"""定义模型"""
def linreg(X,w,b):
    return X @ w + b

"""定义损失函数"""
def squard_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 /2

"""定义优化算法（随机梯度下降SGD）"""
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for params in params:
            params -= lr *params.grad / batch_size
            params.grad.zero_()


"""训练模型"""
lr = 0.025
num_epochs = 10
net = linreg
loss = squard_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)
    
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {w_true - w.reshape(w_true.shape)}')
print(f'b的估计误差: {b_true - b}')