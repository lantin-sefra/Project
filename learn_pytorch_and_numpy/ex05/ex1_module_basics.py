"""
练习 1：自己写一个 nn.Module（对应笔记 ch05_1）
目标：写一个层数可配置的 MLP，不看笔记也能写出骨架。
运行：python ex1_module_basics.py     全部 assert 通过会打印 OK
参考解：solutions/ex1_module_basics.py（卡 15 分钟再看）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMLP(nn.Module):
    """DeepMLP(784, [128, 64], 10)  ->  784 -> 128 -> 64 -> 10，隐藏层后接 ReLU，最后一层不接。"""

    def __init__(self, in_dim, hidden_dims, num_classes):
        # TODO 1: 第一行必须是什么？
        super().__init__()
        # TODO 2: 用 nn.ModuleList 存所有 Linear（为什么不能用普通 list？见 ch05_2）
        self.layer = nn.ModuleList()
        for dims in hidden_dims:
            self.layer.append(nn.Linear(in_dim,dims))
            in_dim = dims    
        self.layer.append(nn.Linear(in_dim,num_classes))
        #   提示：先拼 dims = [in_dim] + hidden_dims + [num_classes]，再两两配对

    def forward(self, x):
        # TODO 3: 除最后一层外，每层后接 F.relu；最后一层直接返回（输出 logits）
        for layer in self.layer[:-1]:
            x = F.relu(layer(x))
        return self.layer[-1](x)


m = DeepMLP(784, [128, 64], 10)
out = m(torch.randn(4, 784))
assert out.shape == (4, 10), f"输出 shape 不对: {out.shape}"

# TODO 4: 先在纸上算出参数总数，再填进来（不要用代码算了再抄）
expected = (784*128 +128) + (128*64 + 64 )+ (64*10 + 10)
n = sum(p.numel() for p in m.parameters())
assert n == expected, f"你算的是 {expected}，实际是 {n}"
assert len(list(m.parameters())) == 6, "3 个 Linear 应有 6 个参数张量（weight+bias）"
print("ex1 OK")
