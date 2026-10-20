# 第 2 章 · Tensor 基础

> **设备提示**：AMD ROCm 用户运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码写 `device="cuda"`；NVIDIA CUDA 写法相同。本章示例默认在 CPU 上即可跑通。

Tensor（张量）是 PyTorch 里的核心数据容器：多维数组 +（可选）自动求导 +（可选）GPU。  
本章目标：会创建、看 shape、切片、变形、拼接，并理解广播与原地操作——写网络时 80% 的 shape 错误出在这里。

📎 理论：

- [d2l · 数据操作](https://zh.d2l.ai/chapter_preliminaries/ndarray.html)
- [PyTorch Tensor 教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- 广播思想同源 NumPy：[Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## 2.1 创建 Tensor

**是什么**：把数字放进 Tensor，供后续运算与求导。

**怎么用**：

```python
import torch
import numpy as np

torch.manual_seed(42)

# 从 Python 嵌套列表创建；整数默认 int64
a = torch.tensor([[1, 2], [3, 4]])
print(a, a.dtype, a.shape)
# tensor([[1, 2],
#         [3, 4]]) torch.int64 torch.Size([2, 2])

# 常用工厂函数
b = torch.zeros(2, 3)     # shape: [2, 3]，默认 float32
c = torch.ones(2, 3)      # shape: [2, 3]
d = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]，shape: [5]
e = torch.randn(2, 3)     # 标准正态 N(0,1)，shape: [2, 3]
print(d, d.shape)
# tensor([0, 2, 4, 6, 8]) torch.Size([5])

# 任意均值/标准差的正态：torch.normal(mean, std, size)
x = torch.normal(0, 1, (4, 2))      # 每个元素 ~ N(0, 1)，造特征常用
noise = torch.normal(0, 0.01, (4,)) # 小方差噪声，贴在 0 附近
print(x.shape, noise.shape)
# torch.Size([4, 2]) torch.Size([4])

# 与 NumPy 共享内存（慎用）
n = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(n)   # dtype 跟随 numpy，这里是 float64
print(t, t.dtype)
# tensor([1., 2., 3.], dtype=torch.float64) torch.float64
n[0] = 99
print(t)                  # 一起变了！
# tensor([99.,  2.,  3.], dtype=torch.float64)
```

其他常用创建（了解即可）：

| API | 作用 |
|-----|------|
| `torch.rand(*size)` | 均匀 `[0,1)` |
| `torch.randn(*size)` | 标准正态 \(\mathcal{N}(0,1)\)，等价 `torch.normal(0, 1, size)` |
| `torch.normal(mean, std, size)` | 正态 \(\mathcal{N}(\mu,\sigma^2)\)；`std` 是**标准差**不是方差 |
| `t.normal_(mean=0, std=1)` | **原地**把已有张量填成正态（带 `_`） |
| `torch.randint(low, high, size)` | 整数随机 |
| `torch.full(size, value)` | 填常数 |
| `torch.eye(n)` | 单位矩阵 |
| `torch.linspace(start, end, steps)` | 等间隔 |
| `torch.zeros_like(x)` / `ones_like(x)` | 同形状 |

`torch.normal` 还可以让每个位置有自己的 \(\mu,\sigma\)（`mean` / `std` 为同形状张量）：

```python
mean = torch.tensor([0., 10.])
std  = torch.tensor([1., 0.01])
print(torch.normal(mean, std))   # 第 0 个大约在 0 附近晃，第 1 个几乎贴着 10
```

**易错点**：

1. `torch.tensor(5)` 是 0 维标量张量；`torch.tensor([5])` 是 1 维。打印长得像，`.shape` 不同。
2. `torch.from_numpy` **共享内存**；要独立副本用 `torch.tensor(n)` 或 `t.clone()`。
3. 列表里是整数 → `int64`；神经网络权重通常要 `float32`，创建时写 `dtype=torch.float32`。
4. `torch.normal` 的第二个参数是标准差 \(\sigma\)；方差 \(\sigma^2=0.01\) 应对 `std=0.1`。d2l 写 `torch.normal(0, 0.01, y.shape)` 是「很小的噪声」，不是方差 0.01。
5. `t.normal_()` 是原地写，会改 `t` 本身；要新张量用 `torch.normal(...)` / `torch.randn`。叶子且 `requires_grad=True` 时原地填充会报错（见 2.7）。

**什么时候用**：构造输入、标签、掩码；人造数据 `X ~ N(0,1)`、观测噪声 `ε ~ N(0,σ)`；随机初始化权重（更常见的是 `nn.Linear` 内部初始化，见第 5 章）。线性回归从零实现里的 `synthetic_data` 就是 `normal` 造 `X` 再造噪声。

**与 NumPy**：`np.zeros` ↔ `torch.zeros`，`np.arange` ↔ `torch.arange`，几乎一一对应。

---

## 2.2 `dtype`、`shape`、`ndim`、`numel`

**是什么**：读懂一个张量「什么类型、几维、多大」。

**怎么用**：

```python
import torch

x = torch.randn(2, 3, 4)
print(x.shape)   # torch.Size([2, 3, 4])
print(x.ndim)    # 3
print(x.numel()) # 24  （2*3*4 个元素）
print(x.dtype)   # torch.float32  （randn 默认）

# 改类型（返回新张量，除非已是目标类型）
y = x.to(torch.float64)
print(y.dtype)   # torch.float64

z = torch.tensor([1, 2], dtype=torch.float32)
print(z.dtype)   # torch.float32
```

常见 `dtype`：

| dtype | 用途 |
|-------|------|
| `torch.float32` | 训练默认（权重、激活） |
| `torch.float16` / `bfloat16` | 混合精度（进阶） |
| `torch.int64` | 分类标签（`CrossEntropyLoss` 要求） |
| `torch.bool` | 掩码 |

**易错点**：

1. 标签误用 `float32` → `CrossEntropyLoss` 报错或行为诡异；分类标签用 `long`（`int64`）。
2. `x.shape[0]` 是 batch 维时，别和特征维搞混——**养成打印 shape 的习惯**。
3. `.size()` 与 `.shape` 等价；更推荐 `.shape`。

**什么时候用**：每次写完一行张量运算，立刻 `print(t.shape)`，比事后查 `RuntimeError` 快十倍。

- 理论指导： [d2l-张量基础](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html#id7)
---

## 2.3 索引与切片

**是什么**：取出子张量或单个元素；语法接近 NumPy。

**怎么用**：

```python
import torch

# x: [3, 4]
x = torch.arange(12).reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(x[1], x[1].shape)           # 第 1 行，shape: [4]
# tensor([4, 5, 6, 7]) torch.Size([4])

print(x[:, 1], x[:, 1].shape)     # 第 1 列，shape: [3]
# tensor([1, 5, 9]) torch.Size([3])

print(x[1, 2])                    # 标量张量 6（0 维）
# tensor(6)

print(x[0:2, 1:3])                # shape: [2, 2]
# tensor([[1, 2],
#         [5, 6]])                #左闭右开

# 布尔索引：结果变为 1 维
mask = x > 5
print(x[mask])
# tensor([ 6,  7,  8,  9, 10, 11])

# 整数数组索引（按行取）
idx = torch.tensor([0, 2])
print(x[idx])                     # shape: [2, 4]
# tensor([[ 0,  1,  2,  3],
#         [ 8,  9, 10, 11]])

#推论：idx 多一维，结果就在「样本维」前面多一维格子
idx = torch.tensor([[0,2],[0,2]])
print(x[idx])                     # shape: [2,2,4]
#tensor([[[ 0,  1,  2,  3],
#         [ 8,  9, 10, 11]],
#
#        [[ 0,  1,  2,  3],
#         [ 8,  9, 10, 11]]]) torch.Size([2, 2, 4])
```

### 用张量当下标：不是套娃，是行号清单

`features[batch_indices]` **不是**把小张量嵌进大张量，而是用 `batch_indices` 里的整数当**第 0 维的行号**，按这个顺序把对应样本抽出来。和 `a[0]`、`a[0:3]` 同类，只是下标从「一个 int / 一段连续切片」换成「一排号码」。

```python
features = torch.tensor([
    [10., 11.],   # 行 0
    [20., 21.],   # 行 1
    [30., 31.],   # 行 2
    [40., 41.],   # 行 3
])
idx = torch.tensor([3, 0, 2])     # 一维、整型，值=行号
print(features[idx])
# tensor([[40., 41.],
#         [10., 11.],
#         [30., 31.]])            # shape [3, 2]，顺序跟 idx
```

`labels[idx]` 必须用**同一串**下标，`(X, y)` 才配对。d2l 的 `data_iter` 就是：把 `0..N-1` shuffle 再按批切成一维 `batch_indices`，然后 `yield features[idx], labels[idx]`。`[0:10]` 只能连续取；小批量要随机顺序，所以必须用整数数组索引。

**`idx` 的维数 ≠ 数据的维数。** `idx` 只决定抽出的样本怎么排列；每个样本长什么样，由 `features` 剩下的维跟过来。`data_iter` 无论特征是向量还是图，行号清单都是**一维** `(B,)`。

| `features` | `idx` | 结果 |
|------------|-------|------|
| `(N, d)` 每行一个向量 | `(B,)` | `(B, d)` |
| `(N, H, W)` 每份一个矩阵 | `(B,)` | `(B, H, W)` |
| `(N, C, H, W)` 每份一张图 | `(B,)` | `(B, C, H, W)` |
| 任意 | 标量 `3` | 去掉第 0 维，剩下那一个样本 |
| 任意 | 二维 `(a, b)` | 多一个格子维：`(a, b, …)`（上面 `x[idx]` 的 `[2,2,4]`） |

不要推论成「数据是三维，下标就要二维去抽矩阵」。三维 `(N,H,W)` 抽一批矩阵，仍然是一维 `idx:(B,)` → `(B,H,W)`。

取出 Python 数字用 `.item()`（仅限单元素）：

```python
v = x[1, 2].item()   # 6，类型是 Python int
```

**易错点**：

1. `x[1]` 会**降维**（去掉被索引的那一维）；若要保持二维可写 `x[1:2]`。
2. 切片是**视图**（多数情况共享存储）；改切片可能改原张量。要副本用 `.clone()`。
3. 布尔索引后 shape 变 1 维，不能想当然还当矩阵用。
4. `[start: stop]` 是左闭右开 `[start, stop)`的，和 Python / NumPy 一样。
5. `arange` 后，返回一个一维的张量，如果你在同名张量上修改，维度会消失。
6. `features[tensor]` 里那个 tensor 是行号，不是嵌套结构。特征和标签要用同一 `idx`。
7. 整数数组索引作用在**被索引的那一维**上（这里是 dim0）；`idx` 升维只是把选中的样本排成格子，不会自动改成「抽矩阵要用二维下标」。

**什么时候用**：取 batch 中某条样本、截序列、做 mask、按类别收集样本；`data_iter` / `DataLoader` 按随机下标组小批量。

---

## 2.4 `view` / `reshape` / `permute` / `transpose` / `squeeze` / `unsqueeze`

**是什么**：不改（或很少改）数据内容，只改「怎么看这些数」——训练代码里最高频的操作之一。

### reshape / view

```python
import torch

x = torch.arange(12)              # shape: [12]
print(x.view(3, 4))               # 要求内存连续
print(x.reshape(3, 4))            # 更宽松，不连续时可能拷贝
print(x.reshape(3, -1))           # -1 自动推算为 4
# 三者打印结果相同，均为 [3, 4] 的 0..11
```

**`view` vs `reshape`（必记）**：

```python
t = torch.arange(12).reshape(3, 4).t()   # 转置后通常不连续
print(t.is_contiguous())                 # False
# t.view(12)  -> RuntimeError: view size is not compatible ...
print(t.reshape(12))                     # 可以，可能发生拷贝
# tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
```

新手建议：**一律用 `reshape`**；只有明确需要视图且保证连续时才用 `view`。

### permute / transpose

```python
a = torch.randn(2, 3, 4)          # [B, H, W] 之类
print(a.permute(0, 2, 1).shape)   # [2, 4, 3]  任意重排维度
print(a.transpose(1, 2).shape)    # [2, 4, 3]  只交换两维
```

图像常见：`[B, H, W, C]` → `[B, C, H, W]`：

```python
# img: [B, H, W, C] -> [B, C, H, W]
# img = img.permute(0, 3, 1, 2)
```

### squeeze / unsqueeze

```python
b = torch.zeros(1, 3, 1, 4)
print(b.squeeze().shape)      # [3, 4]   去掉所有长度为 1 的维
print(b.squeeze(0).shape)     # [3, 1, 4] 只去掉 dim0

c = torch.zeros(3, 4)
print(c.unsqueeze(0).shape)   # [1, 3, 4]  常用来加 batch 维
print(c.unsqueeze(1).shape)   # [3, 1, 4]
```

**易错点**：

1. `view` 在非连续张量上报错——转置 / `permute` 之后尤其常见。
2. `squeeze()` 无参会去掉**所有** 1 维，batch=1 时可能把 batch 维也挤掉，训练时很危险；**指定 dim** 更安全。
3. `reshape` 只改形状，元素总数必须不变：`2*3*4` 不能 `reshape` 成 `[5, 5]`。

**什么时候用**：

| 需求 | API |
|------|-----|
| 展平特征 | `x.reshape(B, -1)` |
| NHWC↔NCHW | `permute` |
| 矩阵转置 | `t` / `transpose` |
| 加/减 batch 维 | `unsqueeze` / `squeeze(dim)` |

📎 理论：行优先存储与 stride 概念见 [PyTorch Internals（可选）](http://blog.ezyang.com/2019/05/pytorch-internals/)；初学只需知道「不连续时 view 可能失败」。

---

## 2.5 拼接与拆分：`cat` / `stack` / `split` / `chunk`

**是什么**：把多个张量拼起来，或把一个张量切开。

**怎么用**：

```python
import torch

a = torch.ones(2, 3)    # [2, 3]
b = torch.zeros(2, 3)   # [2, 3]

# cat：沿已有维度拼接（不新增维）
print(torch.cat([a, b], dim=0).shape)  # [4, 3]
print(torch.cat([a, b], dim=1).shape)  # [2, 6]

# stack：新建一个维度再堆叠，要求形状相同
print(torch.stack([a, b], dim=0).shape)  # [2, 2, 3]  新维在最前
print(torch.stack([a, b], dim=1).shape)  # [2, 2, 3]  新维在中间
print(torch.stack([a, b], dim=2).shape)  # [2, 3, 2]  新维在最后
```

> `stack(..., dim=0)` 与 `dim=1` 在本例中 shape 都是 `[2,2,3]`，但语义不同：前者是「两张 `[2,3]` 叠成批次」，后者是「在原 dim0 与 dim1 之间插入堆叠维」。不确定时打印一个具体值核对。
但是，请思考一下，stack 是用什么逻辑把数据组织起来的？

```python
x = torch.arange(12).reshape(3, 4)   # [3, 4]

# split：按「每段长度」切
parts = torch.split(x, 2, dim=1)     # 沿列，每段宽 2
print([p.shape for p in parts])
# [torch.Size([3, 2]), torch.Size([3, 2])]

# chunk：尽量均分成 n 块（最后一块可能更短）
chunks = torch.chunk(x, 2, dim=0)
print([p.shape for p in chunks])
# [torch.Size([2, 4]), torch.Size([1, 4])]
```

**易错点**：

1. 把 `cat` 和 `stack` 用混：要「加一维组成 batch」用 `stack`；要「加长某一维」用 `cat`。
2. `cat` 要求除拼接维以外的 shape **完全一致**。
3. `chunk` 不能整除时最后一块更短，循环里别写死每块大小。

**什么时候用**：多卡拼 batch、多模态特征拼接、按时间维切开序列。

---

## 2.6 广播机制（Broadcasting）

**是什么**：形状不同但「可对齐」的张量做逐元素运算时，框架在长度为 1 的轴上**逻辑复制**（不真的 `repeat`），使两边形状一致后再算。

📎 理论：[NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)（PyTorch 与之相同，先读这一页的示意图）。

**对齐规则（从最后一维往左比）**：把两个 `shape` 右端对齐，缺的维补在**左边**，当成 1。每一维兼容当且仅当：

1. 两维相等，或
2. 其中一方为 1（可拉长到另一方）

否则报错。结果在每一维取两边的最大值。

先把 shape 写出来右对齐，比盯着「行/列」可靠。一维张量 `shape (k,)` **既不是行也不是列**，右对齐时只占**最后一维**：

```text
A     (2, 3)
b         (3)     # 左边缺一维 → 当成 (1, 3)，沿 dim0 复制
A + b (2, 3)

A     (2, 3)
c     (2, 1)      # 最后一维 1 → 拉成 3，沿 dim1 复制
A + c (2, 3)

A     (2, 3)
d        (2)      # 当成 (1, 2)；3 vs 2 且都不是 1 → 报错
```

**怎么用**：

```python
import torch

a = torch.ones(2, 3)                          # [2, 3]
b = torch.arange(3).float()                   # [3]  → 当成 [1, 3]
print(a + b)
# tensor([[1., 2., 3.],
#         [1., 2., 3.]])                      # 每一行都加上 [0, 1, 2]

c = torch.arange(2).float().unsqueeze(1)      # [2] → [2, 1]  （[:, None] 同义）
print(c)
# tensor([[0.],
#         [1.]])
print(a + c)
# tensor([[1., 1., 1.],
#         [2., 2., 2.]])                      # 每一列都加上同一个列向量
```

网络里的偏置就是第二种：`W @ X` 是 `(n_hid, p)`（**每列一个样本**）时，`b` 是一维 `(n_hid,)`。若直接加，右对齐变成 `(n_hid, p) + (p,)`，`p` 对不上 `n_hid` 会报错。先写成列再广播：

```python
WX = torch.ones(2, 3)                         # 2 个神经元 × 3 个样本
b  = torch.tensor([10., 20.])                 # (2,) 一维，还不是列
print(WX + b[:, None])                        # (2, 1) 沿样本维铺开 = b 1^T
# tensor([[11., 11., 11.],
#         [21., 21., 21.]])
```

`nn.Linear` 默认**每行一个样本** `(B, out)`，偏置 `(out,)` 右对齐正好对上最后一维，所以不必 `[:, None]`。约定一变，缺的那一维就换边了。

### 一维没有行/列：`+` 和 `@` 补 1 的方向相反（必记）

`shape (k,)` **既不是行向量也不是列向量**。课本里向量总是 \(k\times 1\)；PyTorch / NumPy 的一维没有那根轴。运算开始时框架才**临时补一根长度为 1 的轴**——补在哪边，由运算符决定。两套规则不一致，不是记错。

| 运算 | `(k,)` 临时当成 | 1 补在哪 | 乘/加完 |
|------|-----------------|----------|---------|
| `+` `*` 等广播 | 只占**最后一维**，像行 `(1, k)` | 缺的维补在**左边**（右对齐） | 保持广播后的形状 |
| 右边 `@`：`A @ v` | **列** `(k, 1)` | 右边 | **挤掉**这根 1 → 结果少一维 |
| 左边 `@`：`v @ A` | **行** `(1, k)` | 左边 | 同样挤掉 1 |
| `u @ v` 两个一维 | 点积：`(1,k)@(k,1)` | 两侧各补 | 得到**标量**（0 维） |

```text
广播 (2, 3) + (3,)
  (2, 3)
     (3)     → 当成 (1, 3)  → 结果 (2, 3)

矩阵乘 (2, 3) @ (3,)
  (2, 3) @ (3, 1)   临时当成列
              挤掉 1  → 结果 (2,)     ← 数值像乘列向量，形状比课本少一维
```

所以会觉得拧：广播时一维从最右侧对齐（像行），`A @ v` 又按列来乘。其实是两个运算符。看见 `@` 问「收缩轴能不能对上」；看见 `+` 在纸上**右对齐**。不要用加偏置的 `[:, None]` 去套 `X @ w`。

乱了就自己写成二维，和数学一致：

```python
w = w.reshape(-1, 1)   # (d, 1) 真正的列
y = X @ w              # (n, d) @ (d, 1) → (n, 1)
```

`synthetic_data` 里 `(n, d) @ (d,)` 得到 `(n,)`，最后 `y.reshape((-1, 1))` 就是在补回 `@` 挤掉的那一维。

**例：列样本加偏置 vs 行样本乘一维 `w`**

| | 两层网络 `WX + b` | d2l `X @ w + b` |
|--|--|--|
| 样本在哪 | 列：`WX` 为 `(n_hid, p)` | 行：`X` 为 `(n, d)` |
| 一维出现在 | **加法**里的 `b:(n_hid,)` | **乘法**里的 `w:(d,)` |
| 要不要 `[:, None]` | 要：`(n_hid, p)+(n_hid,)` 右对齐失败 | 不要：`@` 不是广播 |
| `b` 是标量 `4.2` | — | Python `float` **不能** `b[:, None]`（`not subscriptable`）；标量加到任何形状都合法 |

**易错点**：

1. 把一维当成列向量：`(m, n) + (m,)` 通常**不能**广播；要沿列复制用 `(m, 1)`（`unsqueeze(1)` / `[:, None]`）。
2. 广播「能算」不等于「语义对」。`(B, 1) + (B,)` 会把 `(B,)` 当成最后一维，可能 silently 算错。意图用 `unsqueeze` 写出来。
3. 调试：打印双方 `shape`，在纸上右对齐逐维看。
4. `(n, d) @ (d,)` 能算，不代表 `(n, d) + (d,)` 规则相同。前者矩阵–向量乘且结果是 `(n,)`；后者才是广播。`[:, None]` 只服务加法对齐。
5. `b[:, None]` 只能用在 **Tensor** 上。Python `float` 没有下标。

**什么时候用**：加偏置、加位置编码、mask 做成 `[B,1,1,T]` 去乘注意力分数等。

---

## 2.7 原地操作（in-place）

**是什么**：名字带 `_` 后缀的方法，直接改当前张量的存储，例如 `add_`、`mul_`、`zero_`、`copy_`。

**怎么用**：

```python
import torch

x = torch.ones(2, 2)
y = x            # y 与 x 指向同一存储
x.add_(1)        # 原地 +1
print(x)
# tensor([[2., 2.],
#         [2., 2.]])
print(y)         # y 也变了！
# tensor([[2., 2.],
#         [2., 2.]])
```

与需要梯度的张量：

```python
x = torch.tensor([1., 2.], requires_grad=True)
# x.add_(1)  -> RuntimeError: a leaf Variable that requires grad
#              is being used in an in-place operation.
```

**易错点**：

1. 原地改共享存储的张量 → 别名变量一起变，极难查的 bug。
2. 对 `requires_grad=True` 的叶子张量原地改 → 直接报错（破坏计算图）。
3. 训练循环里 `optimizer` 更新参数属于框架内部的安全原地写；**你自己写模型逻辑时，优先用非原地** `x = x + 1`。

**什么时候用**：显存极度紧张且你完全清楚别名关系时；或 `model.zero_grad()` / `tensor.zero_()` 这类惯例 API。默认能不用就不用。

---

## 2.8 小结：shape 思维导图

```text
创建 → 看 dtype/shape
      → 索引切片（可能降维）
      → reshape / permute（改看法）
      → cat/stack（改批次结构）
      → 广播（右对齐后逐元素算）
      → @ 时一维按「左行右列」临时补 1，乘完挤掉
```

遇到报错先问三句：

1. 两边 shape 各是什么？当前是 `@` 还是 `+`？（规则不同）
2. 我是否误用了 `view` / `squeeze()` / 把一维当成列？
3. 是否在 GPU/CPU 或 float/long 上不一致？（device/dtype 见第 1 章与第 10 章）

---

## 本章小结

| 主题 | 记住 |
|------|------|
| 创建 | `tensor` / `zeros` / `arange` / `randn` / `normal(μ,σ,size)`；`from_numpy` 共享内存 |
| 元信息 | `shape` / `dtype` / `numel` |
| 索引 | 切片连续取；`x[idx]` 用一维整型当下标抽第 0 维，`idx` 维数≠数据维数 |
| 变形 | 优先 `reshape`；换维用 `permute`；加减维用 `unsqueeze`/`squeeze(dim)` |
| 拼接 | `cat` 不加维，`stack` 加维 |
| 广播 | 从尾维对齐；一维只占最后一维；显式 `unsqueeze` 更清晰 |
| 一维 vs `@` | 一维没有行列；`A @ v` 右边当列且挤掉 1；不确定就 `reshape(-1,1)` |
| 原地 | `_` 后缀有副作用；叶子梯度张量上会报错 |

## 练习

1. 创建 `torch.arange(24)`，依次变成 `[2, 3, 4]`、`[4, 6]`、`[24]`，每步打印 `shape`。
2. 构造 `x` 为 `[2, 3, 4]`，用一次 `permute` 得到 `[2, 4, 3]`，再用 `reshape` 得到 `[2, 12]`。
3. 解释并验证：为何 `torch.stack([a,b], 0)` 与 `torch.cat([a.unsqueeze(0), b.unsqueeze(0)], 0)` 结果相同（`a,b` 形状相同）。
4. 写出三个**不能**广播相加的 shape 对，并实际运行确认报错。
5. 验证并解释：`torch.ones(2, 3) @ torch.ones(3)` 的 shape 为什么是 `[2]` 而不是 `[2, 1]`；再与 `torch.ones(2, 3) @ torch.ones(3, 1)` 对比。
6. （综合）模拟一个 mini-batch：用 `randn` 生成 `images: [8, 1, 28, 28]`，把它变成 MLP 输入 `[8, 784]`；再用 `unsqueeze` 把某一张图变回 `[1, 1, 28, 28]`。
7. 令 `X = torch.arange(24).reshape(4, 2, 3)`，`idx = torch.tensor([3, 0])`。先猜 `X[idx].shape`，再运行；把 `idx` 改成 `[[3, 0], [1, 1]]` 再猜一次。解释：为什么抽「矩阵」仍然用一维 `idx`。
