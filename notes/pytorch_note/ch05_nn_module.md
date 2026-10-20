# 第 5 章 · `torch.nn` 核心（速查手册）

> **这是速查手册，不是教程。** 想学会用，先走 [学习路线与任务表](./ch05_0_roadmap.md)：
> [5.1 Module 慢讲](./ch05_1_module_walkthrough.md) → [5.2 注册与参数](./ch05_2_registration_params.md) → [5.3 Linear](./ch05_3_Linear.md) → [5.4 shape 推导](./ch05_4_shape_drills.md) → [5.5 train/eval 与损失](./ch05_5_train_eval_loss.md)。
> 本页保留完整 API 一览，用来**查**。

> **设备提示**：AMD ROCm 用户运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码写 `device="cuda"`；NVIDIA CUDA 写法相同。本章示例 CPU 即可。

`torch.nn` 把「层、参数、损失函数」封装成可组合的模块。  
本章目标：会写 `nn.Module`，熟悉最常用层的 **shape 变化**，分清 `nn.*` 与 `nn.functional`，并为第 9 章的 MLP/CNN 打好底。

📎 理论：

- [d2l · 多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- [d2l · 卷积神经网路](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)
- [d2l · 循环神经网路](https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html)
- [PyTorch Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- BatchNorm：[Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)（了解动机即可）

---



## 5.1 `nn.Module`：`__init__` / `forward`

**是什么**：所有网络的基类。在 `__init__` 里注册子层与参数，在 `forward` 里描述前向计算。

**怎么用**：

- 写法骨架：`super().__init__()` + `forward`，调用用 `model(x)`（慢讲见 [5.1](./ch05_1_module_walkthrough.md)）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, num_classes=10):
        super().__init__()                  # 必须先调：否则登记表没建好
        self.fc1 = nn.Linear(in_dim, hidden)# 自动注册参数
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: [B, 784] -> [B, 10]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
x = torch.randn(4, 784)
logits = model(x)          # 等价于 model.forward(x)，请用 model(x)
print(logits.shape)
# torch.Size([4, 10])
```

**易错点/注意**：

1. `super().__init__()` 必须在给 `self` 赋子模块之前。忘了会直接报 `AttributeError: cannot assign module before Module.__init__() call`。
2. `forward` 必须自己写；但调用时写 `model(x)`，不要写 `model.forward(x)`（前者会经过钩子等准备工作）。
3. 层要在 `__init__` 里创建并赋给 `self`。若在 `forward` 里每次现建 `nn.Linear`，每次调用都是新的随机参数，且不会出现在 `parameters()` 里；用普通 Python list 存层同理，参数同样不会被优化器看到（见 [5.2](./ch05_2_registration_params.md)）。

**什么时候用**：任何自定义网络；第 9 章的 MLP / CNN 都长这样。

---



## 5.2 参数管理：`parameters` / `state_dict`

**是什么**：查看、统计、保存模型里的可学习张量。

**怎么用**：

```python
# 接上一节的 model = MLP()
n = sum(p.numel() for p in model.parameters())
print(n)   # 101770 = 784*128+128 + 128*10+10

for name, p in model.named_parameters():
    print(name, tuple(p.shape))
# fc1.weight (128, 784)
# fc1.bias   (128,)
# fc2.weight (10, 128)
# fc2.bias   (10,)

sd = model.state_dict()          # 有序字典：名字 -> Tensor（含 BN 的 running 统计）
print(list(sd.keys()))
# ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
```


| API                                  | 用途                                             |
| ------------------------------------ | ---------------------------------------------- |
| `parameters()`                       | 交给优化器：`optim.Adam(model.parameters(), lr=...)` |
| `named_parameters()`                 | 调试、按名冻结                                        |
| `state_dict()` / `load_state_dict()` | 保存与加载权重（第 8 章详述）                               |


冻结某层示例：

```python
for p in model.fc1.parameters():
    p.requires_grad = False
```

**易错点**：

1. `state_dict` 比 `parameters` 更「全」——含 BatchNorm 的 `running_mean` 等缓冲区。
2. 保存整模型 `torch.save(model, path)` 可能绑死类定义路径；**推荐只存** `state_dict`。
3. `numel()` 是元素个数，不是层数。

**什么时候用**：建优化器、看模型大小、存盘、部分微调。

---



## 5.3 常用层（重点看 shape）



### 5.3.1 `nn.Linear`

**是什么**：全连接层 $y = x W^\top + b$。

```python
lin = nn.Linear(20, 10)       # in_features=20, out_features=10
x = torch.randn(32, 20)       # [B, 20]
print(lin(x).shape)           # [32, 10]
print(lin.weight.shape)       # [10, 20]  注意：先行后列存的是 out × in
print(lin.bias.shape)         # [10]
```
也支持更高维最后一维作特征：`[B, T, 20] -> [B, T, 10]`。

**构造**
```
nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```
**关于Shape和Attributes** [[ch05_3_Linear.md]]

**易错点**：`weight` shape 是 `[out, in]`，手写 `@` 时用 `x @ W.T`；直接 `nn.Linear` 不用自己转。


📎 理论：[d2l · Softmax 回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html)

---



### 5.3.2 `nn.Conv2d` 与池化

**是什么**：在 `[B, C, H, W]` 上做局部卷积；CNN 主线的核心。

```python
conv = nn.Conv2d(
    in_channels=1, out_channels=8,
    kernel_size=3, stride=1, padding=1,   # padding=1 + k=3 => H,W 不变
)
pool = nn.MaxPool2d(kernel_size=2)       # H,W 约减半

x = torch.randn(4, 1, 28, 28)            # MNIST 风格
h = conv(x)
print(h.shape)                           # [4, 8, 28, 28]
h = pool(h)
print(h.shape)                           # [4, 8, 14, 14]
```

输出空间尺寸公式（正方形核、等高宽 padding/stride）：

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

```python
conv2 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0)
x = torch.randn(2, 3, 32, 32)
print(conv2(x).shape)                    # [2, 16, 14, 14]
# (32 + 0 - 5) / 2 + 1 = 14
```

常用：`nn.AvgPool2d`、`nn.AdaptiveAvgPool2d((1, 1))`（任意 H×W 压成 1×1，便于接全连接）。

**易错点**：

1. 通道维放错：PyTorch 默认 **NCHW**，不是 NHWC。
2. `padding` / `stride` 算错导致后面 `Linear` 的 `in_features` 对不上——先 `print(h.shape)` 再算展平长度。
3. `kernel_size=3, padding=0` 时每边会掉 2 个像素。

📎 理论：[d2l · 卷积的互相关运算](https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)

---



### 5.3.3 `BatchNorm` / `LayerNorm`

**是什么**：稳定训练的归一化层。CNN 常用 `BatchNorm2d`；Transformer / NLP 常用 `LayerNorm`。

```python
bn = nn.BatchNorm2d(8)                   # 参数 = 通道数
x = torch.randn(4, 8, 14, 14)
print(bn(x).shape)                       # [4, 8, 14, 14]

ln = nn.LayerNorm(16)                    # 归一化最后一维大小
x = torch.randn(4, 10, 16)               # [B, T, C]
print(ln(x).shape)                       # [4, 10, 16]
```


|              | BatchNorm         | LayerNorm               |
| ------------ | ----------------- | ----------------------- |
| 统计范围         | 跨 batch（与 H×W）    | 跨特征维（单样本内）              |
| 对 batch size | 敏感；推理用 running 统计 | 不依赖 batch               |
| 典型场景         | CNN               | RNN / Transformer / MLP |


**易错点**：

1. `model.train()` / `model.eval()` 对 BN 行为影响巨大：训练用当前 batch 统计，推理用滑动平均。
2. `BatchNorm1d` 输入是 `[N, C]` 或 `[N, C, L]`，别和 2d 搞混。
3. batch size=1 时训练模式的 BN 不稳定；小 batch 可考虑 GN/LN。

---



### 5.3.4 `nn.Dropout`

**是什么**：训练时随机置零神经元，减轻过拟合；推理时关闭。

```python
drop = nn.Dropout(p=0.5)
x = torch.ones(2, 8)

drop.train()
print(drop(x))
# 约一半为 0，其余为 2（inverted dropout 缩放）

drop.eval()
print(drop(x))
# 全 1，不再丢弃
```

**易错点**：验证前忘记 `model.eval()` → Dropout 仍在丢弃，指标偏差。  
**什么时候用**：全连接后较常见；现代 CNN 更常用 BN，Dropout 比例别一上来就 0.5。

---



### 5.3.5 `nn.Embedding`

**是什么**：把整数索引映射成稠密向量（词表 / 类别 ID）。

```python
emb = nn.Embedding(num_embeddings=1000, embedding_dim=64)
idx = torch.tensor([[1, 2, 3], [4, 5, 6]])   # [B, T]，元素 ∈ [0, 999]
print(emb(idx).shape)                        # [2, 3, 64]：在末尾多出一个 embedding_dim 维
# torch.Size([2, 3, 64])
```

**易错点**：索引越界；标签若是 `float` 需先转 `long`。padding 可用 `padding_idx=0`。

---



### 5.3.6 `nn.RNN` / `nn.LSTM`（够用即可）

**是什么**：处理序列；组里若做调度轨迹 / 时序，可能遇到。新手优先掌握 LSTM 的输入输出 shape。

```python
lstm = nn.LSTM(
    input_size=32, hidden_size=64,
    num_layers=1, batch_first=True,      # True => 输入 [B, T, C]
)
x = torch.randn(8, 20, 32)               # [B, T, input_size]
out, (h, c) = lstm(x)
print(out.shape)                         # [8, 20, 64]  每步隐状态
print(h.shape, c.shape)                  # [1, 8, 64]   [num_layers, B, H]
```

**易错点**：

1. 默认 `batch_first=False` 时输入是 `[T, B, C]`——强烈建议显式 `batch_first=True`。
2. 分类任务常用最后时间步 `out[:, -1, :]`，或 `h[-1]`。
3. 双向 LSTM 时 `out` 最后一维是 `2*hidden_size`。

📎 理论：[d2l · LSTM](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)

---



### 5.3.7 激活函数

```python
x = torch.tensor([[-1., 0., 2.]])
print(F.relu(x))                  # [[0, 0, 2]]
print(F.gelu(x))                  # 约 [[-0.1587, 0, 1.9545]]
print(torch.softmax(x, dim=-1))   # 沿 dim 归一化成概率
# tensor([[0.0420, 0.1142, 0.8438]])
```


| 激活        | 常见场景                                            |
| --------- | ----------------------------------------------- |
| `ReLU`    | CNN / MLP 默认                                    |
| `GELU`    | Transformer                                     |
| `Sigmoid` | 独立多标签（配合 BCE）                                   |
| `Softmax` | 多分类概率；**与** `CrossEntropyLoss` **不要叠着用**（见 5.5） |


**易错点**：`softmax` 必须指定 `dim`；接 `CrossEntropyLoss` 时网络应输出 **logits**，不要先 Softmax。

---



## 5.4 容器：`Sequential` / `ModuleList` / `ModuleDict`

**是什么**：组织子模块的方式。

```python
# 1) Sequential：按顺序执行，适合无分支的直筒结构
seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)
print(seq(torch.randn(2, 10)).shape)     # [2, 5]

# 2) ModuleList：可在 forward 里自定义循环 / 残差
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

print(Net()(torch.randn(2, 10)).shape)   # [2, 10]
```

**千万别用普通 list 存层**：

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(2)]  # 不会注册！

print(list(Bad().parameters()))          # []  空的
```

`ModuleDict` 按名字取子模块，适合多分支（编码器名 / 任务头名）。

**易错点**：`Sequential` 里难写残差；有 skip connection 时用 `ModuleList` + 手写 `forward`。

---



## 5.5 损失函数

**是什么**：标量目标，接到第 4 章的 `backward`。

```python
# 多分类：输入 logits [B, C]，标签 long [B]
logits = torch.randn(4, 10)
target = torch.tensor([1, 3, 0, 7])       # int64
loss = nn.CrossEntropyLoss()(logits, target)
print(loss)                              # 标量，随机 logits 下约 2.3（≈ ln 10）

# 回归
pred = torch.randn(4, 1)
y = torch.randn(4, 1)
print(nn.MSELoss()(pred, y))

# 二分类：输入 logits [B,1] 或 [B]，标签 float
logits = torch.randn(4, 1)
y = torch.tensor([[1.], [0.], [1.], [0.]])
print(nn.BCEWithLogitsLoss()(logits, y))
```


| 损失                  | 输入             | 标签            |
| ------------------- | -------------- | ------------- |
| `CrossEntropyLoss`  | logits `[B,C]` | `long` `[B]`  |
| `MSELoss`           | 同 shape        | 同 shape float |
| `BCEWithLogitsLoss` | logits         | float 0/1     |
| `L1Loss`            | 同 shape        | 同 shape       |


内部关系（便于理解）：

```text
CrossEntropyLoss(logits, y)  ≡  NLLLoss(LogSoftmax(logits), y)
BCEWithLogitsLoss            ≡  BCELoss(Sigmoid(logits), y)
```

**易错点**：

1. `CrossEntropyLoss` 前又加 `Softmax` → 相当于做了两次，结果错误。
2. 标签写成 `float` 或 one-hot → 报错或需换损失。
3. 忽略 `reduction`：默认对 batch **求平均**。

📎 理论：[d2l · 损失](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function)（交叉熵与最大似然）

---



## 5.6 `nn.Module` vs `nn.functional`

**是什么**：同一运算常有「层对象」和「函数」两种 API。

```python
x = torch.randn(4, 20)
W = torch.randn(10, 20)
b = torch.randn(10)
print(F.linear(x, W, b).shape)           # [4, 10]  函数式，W/b 你自己管

# 对比：nn.Linear 内部持有 Parameter，会进 state_dict / parameters()
```


|     | `nn.Conv2d` / `nn.Dropout`…    | `F.conv2d` / `F.dropout`… |
| --- | ------------------------------ | ------------------------- |
| 参数  | 层内自动持有                         | 你传入 weight                |
| 状态  | BN/Dropout 跟随 `train/eval`     | Dropout 要手动传 `training=`  |
| 推荐  | 有参数或依赖 train/eval 的 → 用 `nn.*` | 无状态激活（ReLU）两者都行           |


实践建议：

- **有参数的层**（Linear/Conv/BN/Embedding）：用 `nn.*`，放在 `__init__`
- **无参数激活**：`F.relu` 或 `nn.ReLU()` 均可；`Sequential` 里用 `nn.ReLU()` 更方便

---



## 5.7 小 CNN 模板（为第 9 章主线预埋）

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B,1,28,28] -> [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [B,32,14,14]
            nn.Conv2d(32, 64, 3, padding=1), # -> [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [B,64,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, 28, 28] -> logits: [B, 10]
        return self.classifier(self.features(x))
```

先把每层 shape 在纸上推一遍，再 `print` 验证——这是写 CNN 的基本功。

---



## 本章小结


| 主题            | 记住                                        |
| ------------- | ----------------------------------------- |
| Module        | `__init__` 注册，`forward` 计算，调用用 `model(x)` |
| 参数            | `parameters` 给优化器；存盘用 `state_dict`        |
| Linear / Conv | 盯住最后一维 / NCHW 与 H_{out} 公式                |
| BN / Dropout  | 必须正确切换 `train()` / `eval()`               |
| 损失            | 分类用 CE + logits + `long` 标签；别先 Softmax    |
| 容器            | 动态层用 `ModuleList`，别用 Python list          |




## 练习

> 以下为原始练习（无自动验收）。推荐改做 `learn_pytorch_and_numpy/ex05/` 里带 `assert` 的版本，见 [任务表](./ch05_0_roadmap.md)。

1. 改写 5.1 的 MLP：加深一层 hidden，统计 `parameters` 总数如何变。
2. 输入 `[8, 3, 32, 32]`，设计两层 `Conv2d`（自定 k/s/p）+ 一次 `MaxPool2d`，使最终特征图为 `[8, 64, 8, 8]`；写出每层 shape。
3. 解释：为何 `CrossEntropyLoss` 的输入不应再接 `Softmax`？写两行代码对比「错误接法」与「正确接法」。
4. 构造一个 `Bad` 网络用 list 存 `Linear`，再构造 `Good` 用 `ModuleList`；打印双方 `list(model.parameters())` 长度。
5. （综合）实现 `SmallCNN`，对 `torch.randn(16, 1, 28, 28)` 前向，确认输出 `[16, 10]`；再与一个同输入展平后的 MLP 对比参数量。

