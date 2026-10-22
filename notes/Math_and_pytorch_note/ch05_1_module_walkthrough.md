# 5.1 `nn.Module` 慢讲：从你写过的 `linreg.py` 出发

> 本篇只做一件事：让你**不看笔记**就能写出一个 `nn.Module`。
> 方法：每个实验都「**先预测 → 再运行 → 对答案**」。预测错了不要紧，错的地方就是你要补的。
> 配套练习：`learn_pytorch_and_numpy/ex05/ex1_module_basics.py`

---

## 0. 起点：你已经会的东西

你手写过线性回归（`learn_pytorch_and_numpy/from_scratch/linreg.py`），后来改成了 `linreg_nn.py`。把两者放一起看，**nn.Module 没有任何新数学**，只是整理代码：

| 手写版（你写过） | Module 版 | 本质 |
|---|---|---|
| `w = torch.zeros(..., requires_grad=True)`、`b = ...` | `nn.Linear(2, 1)` 里自带 | 参数的创建被收进「层」里 |
| `def linreg(X, w, b): return X @ w + b` | `def forward(self, X)` | 前向计算换了个位置 |
| `net(X, w, b)` | `model(X)` | 参数不用每次传，模型自己带着 |
| `sgd([w, b], lr, bs)` | `optim.SGD(model.parameters(), lr)` | 「所有参数」由模型替你收集 |

一句话：**`nn.Module` = 把「参数 + 前向函数」打包成一个对象**，并自动记账（有哪些参数、当前是训练还是推理）。

---

## 1. 一个 Module 做哪三件事

```python
class LinReg(nn.Module):
    def __init__(self):
        super().__init__()                 # ① 开账本
        self.linear = nn.Linear(2, 1)      # ② 登记子层 / 参数（赋值给 self 的瞬间自动登记）

    def forward(self, X):                  # ③ 描述「输入怎么变成输出」
        return self.linear(X)
```

| 步骤 | 代码 | 它在干嘛 | 漏掉会怎样 |
|---|---|---|---|
| ① | `super().__init__()` | 创建内部的登记表（存参数、子模块） | 立刻报 `AttributeError: cannot assign module before Module.__init__() call` |
| ② | `self.xxx = nn.Linear(...)` | 赋值时 Module 偷偷把它记进登记表 | 不用 `self.` 前缀就是局部变量，白建了 |
| ③ | `def forward` | 定义计算 | 调用时报 `NotImplementedError` |

调用时写 `model(x)`，不要写 `model.forward(x)`。`model(x)` 会先做一些准备工作（比如钩子 hook），再去调 `forward`。

> 纠正一个常见说法：忘记 `super().__init__()` **不会**静默出问题，而是直接报错，错误信息很明确。别被吓到。

---

## 2. 逐行拆一个 MLP（慢速）

```python
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)       # 参数: weight [128,784], bias [128]
        self.fc2 = nn.Linear(hidden, num_classes)  # 参数: weight [10,128],  bias [10]

    def forward(self, x):                          # x: [B, 784]
        h = self.fc1(x)                            # h: [B, 128]
        h = F.relu(h)                              # h: [B, 128]   ReLU 不改形状、也没有参数
        return self.fc2(h)                         # out: [B, 10]  logits
```

要点逐条：

1. **`__init__` 的参数是「结构超参数」**（宽度、类别数），不是数据。数据 `x` 只在 `forward` 里出现。
2. **`forward` 里 shape 的流向**要像上面那样逐行标注。一旦你养成习惯，90% 的 shape 报错能自己定位。
3. **`F.relu` 放在 `forward`、不放在 `__init__`**：因为它没有参数，不需要登记。
4. 最后一层**不接激活**：输出 logits，交给损失函数处理（见 5.5）。

### shape 追踪练习（手算）

输入 `x = torch.randn(32, 784)`，填表后再用代码验证：

| 位置 | shape |
|---|---|
| `x` | `[32, 784]` |
| `self.fc1(x)` | ? |
| `F.relu(...)` | ? |
| `self.fc2(...)` | ? |

<details><summary>答案</summary>

`[32,128]` → `[32,128]` → `[32,10]`
</details>

---

## 3. 四个实验（每个都先预测）

### 实验 A：`print(model)` 看到什么？

```python
model = MLP()
print(model)
```

**预测**：会打印几行？每个子层叫什么名字？名字从哪来？

<details><summary>答案</summary>

```text
MLP(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

括号里的名字 `fc1`、`fc2` 就是你写的 `self.fc1`、`self.fc2` 的属性名。`F.relu` 不在里面，因为它没登记。
</details>

**结论**：`print(model)` 是检查「你想要的结构有没有真的建出来」最快的办法。写完模型第一件事就是 print 它。

### 实验 B：梯度从无到有

```python
x = torch.randn(4, 784); y = torch.tensor([1, 2, 3, 4])
print(model.fc1.weight.grad)                       # (a)
loss = F.cross_entropy(model(x), y)
loss.backward()
print(model.fc1.weight.grad.shape)                 # (b)
print(model.fc2.bias.grad.shape)                   # (c)
```

**预测**：(a)(b)(c) 各是什么？

<details><summary>答案</summary>

(a) `None`；(b) `torch.Size([128, 784])`；(c) `torch.Size([10])`。
**梯度和参数同形状**。这也是优化器能直接 `param -= lr * param.grad` 的原因。
</details>

### 实验 C：`.grad` 会累加

```python
m = nn.Linear(3, 1); x = torch.randn(5, 3)
m(x).sum().backward(); g1 = m.bias.grad.clone()
m(x).sum().backward(); print(g1, m.bias.grad)
```

**预测**：第二次 `.grad` 是多少？

<details><summary>答案</summary>

`tensor([5.]) tensor([10.])`。`backward()` 是**累加**到 `.grad` 上，不是覆盖。所以训练循环里要 `optimizer.zero_grad()`。你在 `linreg_nn.py` 里写过这行，现在知道它为什么必须有了。
</details>

### 实验 D：`model.training`

```python
m = MLP()
print(m.training, m.fc1.training)
m.eval()
print(m.training, m.fc1.training)
```

**预测**：输出什么？

<details><summary>答案</summary>

`True True`，然后 `False False`。`eval()` 会**递归**把整棵树的子模块都切成推理模式。对 `Linear` 没有影响，但对 Dropout / BatchNorm 影响很大（见 5.5）。
</details>

---

## 4. 写 Module 的四步清单

每次写新模型，照这个顺序，不用想：

1. **画结构**：纸上写出 `输入 shape → 每层 → 输出 shape`
2. **`__init__`**：`super().__init__()`，然后把**有参数**的层按顺序用 `self.xxx = ...` 建好
3. **`forward`**：按纸上的顺序把层串起来，每行加 shape 注释
4. **验证**：`print(model)` 看结构，`model(torch.randn(...))` 看 shape，数一遍参数量

---

## 小结

- Module = 参数 + `forward`，外加自动记账。
- 三个必写动作：`super().__init__()`、`self.xxx = 层`、`forward`；调用用 `model(x)`。
- 梯度与参数同形状、会累加；`eval()` 递归生效。

## 任务

→ 做 `ex05/ex1_module_basics.py`，通过后进入 [5.2 注册机制与参数管理](./ch05_2_registration_params.md)。
