"""
练习 2：注册机制（对应笔记 ch05_2）
目标：亲手踩一遍「list vs ModuleList」，会冻结层、会数可训练参数。
"""
import torch.nn as nn



class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]


class Good(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 1: 改成能被注册的写法
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

pred_bad = 0
assert len(list(Bad().parameters())) == pred_bad, "Bad 的参数个数和你预测的不一致"
assert len(list(Good().parameters())) == 6, "Good 应有 3 层 × (weight+bias) = 6 个"

g = Good()
# TODO 2: 冻结第 0 层（让它的参数不再训练）
for param in g.layers[0].parameters():
    param.requires_grad = False
# TODO 3: 先用笔算：冻结后可训练参数共多少个元素？
expected_trainable = 220
trainable = sum(p.numel() for p in g.parameters() if p.requires_grad)
assert trainable == expected_trainable, f"实际 {trainable}"

# TODO 4: state_dict 的 key 长什么样？写出集合，如 {"layers.0.weight", ...}
expected_keys = {"layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias"}
assert set(g.state_dict().keys()) == expected_keys
print("ex2 OK")
