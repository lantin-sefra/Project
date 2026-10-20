# 第 3 章 · 矩阵与数学运算

> **设备提示**：AMD ROCm 用户运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码写 `device="cuda"`；NVIDIA CUDA 写法相同。本章示例 CPU 即可。

神经网络里绝大多数计算可以归成三类：**逐元素**、**矩阵乘**、**归约（reduction）**。  
本章目标：看 shape 就能预判结果形状；会在 `@` / `mm` / `bmm` / `einsum` 中选对工具。

📎 理论：

- [d2l · 线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html)
- [mmlbook 第 2 章 Linear Algebra](https://mml-book.github.io/book/mml-book.pdf)（矩阵乘、范数）
- [PyTorch einsum 说明](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- Einstein 求和直觉：[Einops / einsum 教程（可选）](https://rockt.github.io/2018/04/30/einsum)

---

## 3.1 逐元素运算

**是什么**：对对应位置分别计算（加、乘、幂、比较等），要求形状相同或可广播（见第 2.6 节）。

**怎么用**：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])   # [2, 2]
b = torch.tensor([[10., 20.], [30., 40.]])

print(a + b)          # 逐元素加
# tensor([[11., 22.],
#         [33., 44.]])

print(a * b)          # 逐元素乘（不是矩阵乘！）
# tensor([[ 10.,  40.],
#         [ 90., 160.]])

print(a ** 2)         # 等价于 torch.pow(a, 2)
# tensor([[ 1.,  4.],
#         [ 9., 16.]])

print(a > 2)          # 比较 -> Bool 张量，shape 不变
# tensor([[False, False],
#         [ True,  True]])

# 按条件选取(详情见 help(torch.where) )
print(torch.where(a > 2, a, torch.zeros_like(a)))
# tensor([[0., 0.],
#         [3., 4.]])
```

常用还有：`torch.abs` / `sqrt` / `exp` / `log` / `clamp` / `relu`（激活也常写成函数形式，见第 5 章）。

### 掩码（mask）

**是什么**：与数据同形状的「开关图」，用 `True/False` 或 `1/0` 标出哪些位置保留、哪些清零或忽略。比较运算（`>` / `==` 等）的结果就是最常见的 bool 掩码。

**怎么用**（两种等价思路）：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])   # [2, 2]

mask = a > 2                    # Bool，shape 仍是 [2, 2]
# tensor([[False, False],
#         [ True,  True]])

# 思路 A：bool → 0/1，再逐元素乘（「乘开关」）
print(a * mask.float())
# tensor([[0., 0.],
#         [3., 4.]])

# 思路 B：按条件二选一（不必先 .float()）
print(torch.where(mask, a, torch.zeros_like(a)))
# tensor([[0., 0.],
#         [3., 4.]])
```

以后会反复见到的场景：

| 场景 | 掩码含义 |
|------|----------|
| 条件清零 / ReLU 直觉 | 不满足条件的位置 × 0 |
| 变长序列 padding | pad 位置为 0，不算进 loss / attention |
| 损失里忽略某些样本 | `loss * sample_mask` 再求平均 |

**易错点**：

1. **`*` 是逐元素乘，`@` 才是矩阵乘**——这是新手 №1 混淆点。
2. `log` / `sqrt` 遇到负数或 0 会出 `nan`/`-inf`，必要时先 `clamp(min=eps)`。
3. 比较结果是 `bool`；要当 0/1 掩码乘回去，用 `.float()`（或 `.to(a.dtype)`）。`where` 可直接吃 bool，不必先转浮点。
4. 掩码与数据的 shape 须相同或可广播，否则乘不上去（规则同第 2.6 节）。

**什么时候用**：激活、残差相加、掩码筛选、损失里的逐元素项。

**与 NumPy**：运算符语义基本一致（`*` 同样不是矩阵乘）；`np.where` ↔ `torch.where`。

---

## 3.2 矩阵乘：`@` / `matmul` / `mm` / `bmm`

**是什么**：线性层、注意力、几乎所有「特征混合」都靠矩阵乘。

### 二维：`[m, k] @ [k, n] -> [m, n]`

```python
import torch
torch.manual_seed(42)

x = torch.randn(2, 3)    # [B=2, in=3]
W = torch.randn(3, 4)    # [in=3, out=4]

y1 = x @ W
y2 = torch.matmul(x, W)
y3 = torch.mm(x, W)
print(y1.shape)          # torch.Size([2, 4])
# y1、y2、y3 在二维上等价
```

| API | 适用 | 说明 |
|-----|------|------|
| `A @ B` | 最常用 | 语法糖，推荐 |
| `torch.matmul` | 1D~高维 | 会广播 batch 维 |
| `torch.mm` | **仅 2D** | 输入必须都是矩阵 |
| `torch.bmm` | **仅 3D** | `[B,n,m] @ [B,m,p]`，batch 一一对应 |

### 批量三维：`[B, n, k] @ [B, k, m] -> [B, n, m]`

```python
X = torch.randn(8, 2, 3)   # [B, n, k]
Y = torch.randn(8, 3, 4)   # [B, k, m]

print((X @ Y).shape)           # torch.Size([8, 2, 4])
print(torch.bmm(X, Y).shape)   # torch.Size([8, 2, 4])  同样结果
```

### 与向量

```python
v = torch.randn(3)         # [k]
print((x @ v).shape)       # [2, 3] @ [3] -> [2]
```

**易错点**：

1. 左右顺序反了：`W @ x` 若 `W` 是 `[out, in]`、`x` 是 `[B, in]` 会直接 shape 报错——线性层实际是 `x @ W.T` 或 `F.linear(x, W, b)`。
2. `torch.mm` 不能吃 3D：`torch.mm(X[0], Y)` → `RuntimeError`。批量用 `@` / `bmm` / `matmul`。
3. `matmul` 的 batch 广播规则比 `bmm` 宽松，复杂时先画 shape，或改用 `einsum`（更直观）。

**什么时候用**：

- 日常：`@` / `matmul`
- 明确「同 batch 一一相乘、不要乱广播」：`bmm`
- 只要 2D 且想接口严格：`mm`

📎 理论：线性变换 \(y = xW + b\) 见 [d2l · 线性回归的从零开始实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)

---

## 3.3 转置与「最后两维」思维

**是什么**：交换维度，使矩阵乘的「收缩维」对齐。

**怎么用**：

```python
import torch

A = torch.randn(2, 3, 4)     # 例如 [B, n, k]

# 推荐：只转置最后两个维（批量矩阵）
print(A.mT.shape)                 # torch.Size([2, 4, 3])
print(A.transpose(-2, -1).shape)  # 同上

# 任意重排
print(A.permute(0, 2, 1).shape)   # torch.Size([2, 4, 3])
```

**关于 `.T`（重要）**：

对 **二维** 矩阵，`A.T` 仍然常用且清晰。  
对 **高于 2 维** 的张量，`A.T` 会反转**所有**维，且在 PyTorch 2.x 中已标为 deprecated，将来可能直接报错。批量矩阵请用 **`A.mT`** 或 `transpose(-2, -1)`。

```python
# 2D 没问题
M = torch.randn(3, 4)
print(M.T.shape)   # [4, 3]
```

**易错点**：

1. 对 `[B, H, W]` 误用 `.T` 得到 `[W, H, B]`，再去 `@` 会静默算错或报错。
2. 转置后常变得不连续 → 接着 `view` 可能失败（第 2.4 节）；需要时用 `reshape` 或 `.contiguous()`。

**什么时候用**：`x @ W.T`、注意力里 `Q @ K.mT`、把 `[B, T, C]` 变成 `[B, C, T]` 等。

---

## 3.4 归约：`sum` / `mean` / `max` / `argmax` / `norm`

**是什么**：沿某些维「压掉」，得到标量或更低维张量。损失求平均、取类别、算范数都靠它。

**怎么用**：

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(x)
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

print(x.sum())    # 全体求和 -> 标量张量 66.
print(x.mean())   # 5.5

# dim=0：压掉第 0 维（行），对每一列汇总 -> [4]
print(x.sum(dim=0), x.sum(dim=0).shape)
# tensor([12., 15., 18., 21.]) torch.Size([4])

# keepdim=True：保留长度为 1 的维，方便广播
print(x.sum(dim=1, keepdim=True).shape)   # torch.Size([3, 1])
# tensor([[ 6.],
#         [22.],
#         [38.]])
```

### `max` / `argmax`

```python
# 1) 不带 dim：返回全局最大值（标量张量）
print(torch.max(x))                  # tensor(9.)

# 2) 带 dim：沿该维度求最大，返回 (values, indices)
vals, idxs = x.max(dim=1)   # 每行最大值及列下标
print(vals)                 # tensor([ 3.,  7., 11.])
print(idxs)                 # tensor([3, 3, 3])
# 也可以按名字取
out = torch.max(x, dim=0)
print(out.values, out.indices)       # tensor([7., 5., 9.]) tensor([1, 0, 1])

# 只要位置，不要值
print(torch.argmax(x, dim=1))        # tensor([1, 2])，等价于 torch.max(x, dim=1).indices
print(torch.argmax(x, dim=0))        # tensor([1, 0, 1])

# 不指定 dim：先展平成一维再找，返回的是展平后的下标
print(torch.argmax(x))               # tensor(5)，即 x.flatten() 中的位置
```
**keepdim 参数**
```python
print(torch.argmax(x, dim=1).shape)                 # [2]
print(torch.argmax(x, dim=1, keepdim=True).shape)   # [2, 1]，保留被归约的维度
```

分类推理时：`pred = logits.argmax(dim=-1)`，`logits: [B, C] -> pred: [B]`。

### 范数

```python
print(torch.norm(x))            # 全体 L2 范数（约 22.49）
print(torch.norm(x, dim=1))     # 每行 L2 -> [3]
```

**易错点**：

1. **`dim` 搞反**：`dim=i` 表示「沿着 i 消掉」。对 `[B, C]` 在类别维取最大应 `dim=1` 或 `dim=-1`。
2. 丢了 `keepdim=True` 后与原张量广播失败——归一化时常需要 `keepdim`。
3. `x.max()` 无 `dim` 返回标量；`x.max(dim=...)` 返回 **命名元组** `(values, indices)`，不要写成 `m = x.max(1); m + 1` 却期望只对 values 运算。

**什么时候用**：loss 平均、accuracy、LayerNorm 里的均值方差、梯度裁剪的范数等。

📎 理论：范数与距离 [mmlbook §2.5](https://mml-book.github.io/book/mml-book.pdf)；交叉熵与 Softmax 推导见 [d2l · Softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)

---

## 3.5 `einsum` 入门（掌握 5 个模式就够）

**是什么**：`torch.einsum`（爱因斯坦求和约定）。用一个下标字符串方程来描述张量运算，可以替代矩阵乘法、转置、求和、批处理等一大堆操作。复杂张量收缩时比嵌套 `permute + matmul` 更可读。

**读法**：`->` 左边是输入下标，右边是输出下标；**左边出现、右边消失的字母 = 被求和（收缩）的维**。
```python
torch.einsum("输入下标1,输入下标2->输出下标", 张量1, 张量2)
```
1.每个张量的每一维用一个字母标记。
2.出现在输入、但没有出现在输出中的字母，会被求和消去。
3.在多个输入中重复出现的字母，对应维度会按元素相乘（大小必须一致或可广播）。
4.输出下标的字母顺序决定输出的维度顺序。

```python
import torch
torch.manual_seed(0)

A = torch.randn(2, 3)
B = torch.randn(3, 4)

# 1) 矩阵乘：ik,kj->ij（对 k 求和）
print(torch.einsum("ik,kj->ij", A, B).shape)   # [2, 4]，等价于 A @ B

# 2) 转置：ij->ji
print(torch.einsum("ij->ji", A).shape)         # [3, 2]，等价于 A.T

# 3) 求和：全部求和 / 按轴求和
print(torch.einsum("ij->", A))                 # 标量张量，等价于 A.sum()
print(torch.einsum("ij->i", A).shape)          # [2]，对 j 求和，等价于 A.sum(dim=1)
print(torch.einsum("ij->j", A).shape)          # [3]，对 i 求和，等价于 A.sum(dim=0)

# 4) 点积 / 外积（一维向量）
u = torch.randn(3)
v = torch.randn(3)
print(torch.einsum("i,i->", u, v))             # 等价于 torch.dot(u, v)
print(torch.einsum("i,j->ij", u, v).shape)     # [3, 3]，等价于 torch.outer(u, v)

# 5) 逐元素相乘
print(torch.einsum("ij,ij->ij", A, A).shape)   # [2, 3]，等价于 A * A

# 6) 对角线 / 迹
M = torch.randn(4, 4)
print(torch.einsum("ii->i", M).shape)          # [4]，等价于 torch.diagonal(M)
print(torch.einsum("ii->", M))                 # 等价于 torch.trace(M)
```
**批量处理**
```python
# 批量点积：bi,bi->b
x = torch.randn(8, 16)
y = torch.randn(8, 16)
print(torch.einsum("bi,bi->b", x, y).shape)    # [8]

# 注意力分数（简化）：Q @ K^T
# Q, K: [B, L, D] -> scores: [B, L, L]
Q = torch.randn(2, 5, 8)
K = torch.randn(2, 5, 8)
print(torch.einsum("bqd,bkd->bqk", Q, K).shape)  # [2, 5, 5]

# 多头注意力分数：(batch, head, seq_q, dim) x (batch, head, seq_k, dim)
Q = torch.randn(2, 8, 10, 64)
K = torch.randn(2, 8, 12, 64)
scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)   # [2, 8, 10, 12]
# 等价于 Q @ K.transpose(-1, -2)

# 对最后两维做矩阵乘法，前面的维度自动匹配
A = torch.randn(2, 8, 3, 4)
B = torch.randn(2, 8, 4, 5)
out = torch.einsum("...ik,...kj->...ij", A, B)   # [2, 8, 3, 5]
```
再记一个：批量外积 `bi,bj->bij` → `[8, 16, 16]`。

**易错点**：

1. 同一字母必须表示**长度相同**的维，否则报错。
2. 字母只是标签，叫 `i/k` 还是 `b/d` 无所谓，但同一维要用同一字母。
3. 不是所有东西都要上 `einsum`：二维 `x @ W` 写 `@` 更清晰。
4. 调试时先 `print(...shape)` 对照预期输出形状，能快速发现下标写错的问题。

**什么时候用**：多头注意力、双线性、任意「多个张量按维对齐相乘再求和」——读论文公式时特别好用。

---

## 3.6 一张表对照（写代码时速查）

| 你想做的事 | 写法 | 典型 shape |
|------------|------|------------|
| 逐元素乘 | `a * b` | 同形或广播 |
| 矩阵乘 | `a @ b` | `[m,k]@[k,n]->[m,n]` |
| 批量矩阵乘 | `a @ b` / `bmm` | `[B,n,k]@[B,k,m]->[B,n,m]` |
| 转置最后两维 | `a.mT` | `[B,n,k]->[B,k,n]` |
| 按行求和 | `a.sum(dim=1)` | `[B,C]->[B]` |
| 分类预测 | `logits.argmax(dim=-1)` | `[B,C]->[B]` |
| 公式型收缩 | `einsum` | 见上节 |

---

## 本章小结

| 主题 | 记住 |
|------|------|
| `*` vs `@` | 逐元素 vs 矩阵乘 |
| 批量乘 | 优先 `@`；严格 batch 对齐可用 `bmm` |
| 转置 | 2D 用 `.T`；≥3D 批量矩阵用 `.mT` |
| 归约 | 看清 `dim` / `keepdim`；`max(dim=)` 返回值+下标 |
| `einsum` | 先掌握 matmul / 转置 / 求和 / 批量点积 / `bqd,bkd->bqk` |

## 练习

1. 手算并验证：`[4, 5] @ [5, 6]`、`[8, 4, 5] @ [8, 5, 6]` 的输出 shape。
2. 设 `logits = torch.randn(16, 10)`，写出预测类别与「每个样本最大 logit」的代码，打印两者 shape。
3. 用三种写法实现同一矩阵乘：`@`、`matmul`、`einsum('ik,kj->ij')`，用 `torch.allclose` 确认一致。
4. 对 `x: [B, T, C]`，用 `einsum` 算时间维均值，得到 `[B, C]`（等价于 `x.mean(dim=1)`）。
5. （综合）实现朴素线性层前向：`y = x @ W.T + b`，其中 `x:[32,20]`，`W:[10,20]`，`b:[10]`，标注每一步 shape；再用 `torch.nn.functional.linear` 对照结果。
