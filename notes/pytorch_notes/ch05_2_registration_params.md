# 5.2 注册机制与参数管理

> 目标：理解「为什么有的层优化器看得见、有的看不见」，会数参数、冻结层、存取权重。
> 前置：[5.1](./ch05_1_module_walkthrough.md)。配套练习：`ex05/ex2_registration.py`

---

## 1. 心智模型：Module 是一棵树

```text
MLP                      ← 根
├── fc1  (Linear)        ← 子模块
│     ├── weight         ← 参数（叶子）
│     └── bias
└── fc2  (Linear)
      ├── weight
      └── bias
```

- **登记**：在 `__init__` 里执行 `self.xxx = <Module 或 nn.Parameter>` 时，`nn.Module` 重写了赋值，会自动把它挂到树上。
- **遍历**：`model.parameters()` 就是把这棵树的所有 `Parameter` 叶子收集起来。优化器只更新它收集到的参数。

所以**一个层要被训练，只需要满足：它被登记在树上**。其他一切问题都是这条规则的推论。

---

## 2. 实验：list 为什么「不行」，ModuleList 为什么「行」

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]    # 普通 Python list

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

print(len(list(Bad().parameters())))     # ?
print(len(list(Good().parameters())))    # ?
```

**预测**后再看答案：

<details><summary>答案</summary>

`0` 和 `6`。

`Module` 的赋值钩子只认「Module / Parameter / Tensor buffer」。普通 list 对它来说就是个不认识的 Python 对象，原样存进去，**不会往里看**。结果是：模型能前向、能算出 loss，但 `parameters()` 是空的，优化器收到空列表，**训练时什么也不会变**（而且不报错——这是最阴险的 bug）。

`nn.ModuleList` 本身是个 Module，它自己会把里面的每一层再登记一遍。
</details>

**症状自检**：训练时 loss 纹丝不动，先 `print(len(list(model.parameters())))`。

### 容器怎么选

| 容器 | 什么时候用 | 写法 |
|---|---|---|
| `nn.Sequential` | 直筒结构，一层接一层 | `nn.Sequential(a, b, c)`，不用写 forward |
| `nn.ModuleList` | 层数不固定 / 要在 forward 里写循环 / 有残差 | 自己写 forward 去遍历 |
| `nn.ModuleDict` | 多分支、按名字取（多任务头） | `self.heads["cls"](x)` |

---

## 3. 看清树的三个入口

```python
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

a = A()
print([n for n, _ in a.named_modules()])   # ['', 'fc1', 'act', 'fc2']   '' 是根自己
print(list(a.state_dict().keys()))         # ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
```

| API | 回答的问题 |
|---|---|
| `named_modules()` | 树上有哪些节点 |
| `named_parameters()` | 哪些叶子会被训练，叫什么名 |
| `state_dict()` | 要存盘的所有张量（比 parameters 多：含 BN 的 buffer） |

`nn.ReLU()` 在树上有节点（因为是 Module），但它没有参数，所以 `state_dict` 里没有它。

---

## 4. state_dict 的 key = 你起的属性名（重要！）

key 是「路径名」，由你写的 `self.xxx` 决定。

```python
class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)     # 名字和 A 里的 fc1 不同
        self.out = nn.Linear(2, 1)

B().load_state_dict(A().state_dict())
```

**预测**：会成功吗？

<details><summary>答案</summary>

报错：

```text
Error(s) in loading state_dict for B:
    Missing key(s) in state_dict: "hidden.weight", "hidden.bias", "out.weight", "out.bias".
    Unexpected key(s) in state_dict: "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias".
```

**shape 一模一样也没用**，PyTorch 按名字对号入座。所以：存了权重之后，不要随便改 `self.xxx` 的名字。
</details>

`Sequential` 没有属性名，用下标当名字：

```python
s = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
print(list(s.state_dict().keys()))   # ['0.weight', '0.bias', '2.weight', '2.bias']
```

注意 `ReLU` 占了下标 1，所以第二个 Linear 是 `2.` 而不是 `1.`。

---

## 5. 数参数、冻结层

### 数参数量

```python
sum(p.numel() for p in model.parameters())                         # 全部
sum(p.numel() for p in model.parameters() if p.requires_grad)      # 只算可训练
```

`numel()` = 元素个数。`Linear(in, out)` 的参数量 = `in*out + out`。

手算习惯：`MLP(784,128,10)` = `784*128+128 + 128*10+10 = 101770`。

### 冻结

```python
for p in model.fc1.parameters():
    p.requires_grad = False
```

冻结后做一次 backward，检查：

```python
a.fc1.weight.grad     # None：根本没算
a.fc2.weight.grad     # 正常有值
```

训练时建议只把可训练参数交给优化器：

```python
optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
```

---

## 6. `Parameter` 与 `buffer`：一个会训练、一个只是状态

| | 例子 | 出现在 `parameters()` | 出现在 `state_dict()` | 被优化器更新 |
|---|---|---|---|---|
| Parameter | `Linear.weight` | ✅ | ✅ | ✅ |
| buffer | `BatchNorm.running_mean` | ❌ | ✅ | ❌（由前向自己更新） |

```python
bn = nn.BatchNorm1d(3)
print([n for n, _ in bn.named_parameters()])   # ['weight', 'bias']
print(list(bn.state_dict().keys()))
# ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
```

所以存盘用 `state_dict`（不会漏掉 BN 的统计量），建优化器用 `parameters()`。

自己定义时：

- 想要可训练的张量 → `self.scale = nn.Parameter(torch.ones(1))`
- 想要不训练但要随模型走/存盘的张量 → `self.register_buffer("mask", tensor)`

---

## 小结

- 一个层能被训练 ⇔ 被登记在 Module 树上；list 不行，用 `ModuleList`。
- loss 不动的第一反应：数一数 `parameters()` 是不是空的。
- `state_dict` 的 key = 属性名路径，改名会导致加载失败。
- 冻结 = `requires_grad=False`；存盘用 `state_dict`。

## 任务

→ 做 `ex05/ex2_registration.py`，通过后进入 [5.3 Linear](./ch05_3_Linear.md) 和 [5.4 shape 推导](./ch05_4_shape_drills.md)。
