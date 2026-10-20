# 第 4 章 · 自动求导（Autograd）

> **设备提示**：AMD ROCm 用户运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码写 `device="cuda"`；NVIDIA CUDA 写法相同。本章示例 CPU 即可。

自动求导是 PyTorch 相对 NumPy 的核心增量：你写前向公式，框架帮你算 \(\partial L / \partial \theta\)。  
本章目标：搞懂 `requires_grad` → `backward` → `.grad` → `zero_grad` 这条训练主链，并会用 `no_grad` / `detach` 切断图。

📎 理论：

- [d2l · 自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html)
- [PyTorch Autograd 官方教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- 反向传播推导：[CS231n Backprop](https://cs231n.github.io/optimization-2/)（可选）
- 矩阵求导查阅：[The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

---

## 4.1 `requires_grad` 与计算图直觉

**是什么**：给张量打开「需要梯度」开关后，所有基于它的可微运算会被记录成一张**计算图**；`backward` 时沿图反向传播。

**怎么用**：

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x.requires_grad, x.is_leaf)
# True True
```

- **叶子节点（leaf）**：通常是你直接创建、或 `nn.Parameter` 的参数；梯度最终会积在它们的 `.grad` 上。
- **非叶子节点**：中间结果（如 `y = x * 2`）；默认不保留 `.grad`，只保留 `grad_fn` 以便反传。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
print(y)
# tensor(11., grad_fn=<AddBackward0>)   # 有 grad_fn => 不是叶子，参与了图
```

**易错点**：

1. 整数张量不能求导：`torch.tensor(2, requires_grad=True)` 会报错，需浮点。
2. `torch.tensor(np_array)` 默认 `requires_grad=False`；要对参数求导，创建后设 `True`，或用 `nn.Parameter`。
3. 已废弃的 `Variable` 不要再用——直接用 Tensor。

**什么时候用**：模型参数、需要反传的输入（如对抗样本、部分可学习输入）。数据 `x`、标签 `y` 一般**不需要** `requires_grad`。

---

## 4.2 `backward()` 与 `.grad`

**是什么**：对**标量**损失调用 `backward()`，Autograd 把梯度写进各叶子张量的 `.grad`。

**怎么用**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1    # dy/dx = 2x + 3；x=2 时为 7
y.backward()
print(y.item(), x.grad)
# 11.0 tensor(7.)
```

向量先归约成标量再反传（训练里几乎总是这样）：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()        # 标量；∂y/∂x = 2x
y.backward()
print(x.grad)
# tensor([2., 4., 6.])
```

**易错点**：

1. 对非标量直接 `backward()` 会报错（见 4.6）；训练时用 `loss.mean()` / `loss.sum()`。
2. `.grad` 在第一次 `backward` 前是 `None`，不是 0。
3. 默认 `backward` 后计算图会释放；同一图上再次 `backward` 需 `retain_graph=True`（通常训练不需要）。

**什么时候用**：每个训练 step 在算完 `loss` 之后立刻 `loss.backward()`。

**与 NumPy**：NumPy 不算梯度；你要手推或用 JAX 等。PyTorch 把「手推反向」自动化了。

---

## 4.3 梯度累积与 `zero_grad()`

**是什么**：`.grad` **默认累加**，不会在下一次 `backward` 时自动清空。这是故意设计（可用于梯度累积），但常规训练每步必须先清零。

**怎么用**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
for i in range(2):
    y = x ** 2            # dy/dx = 4（在 x=2 时）
    y.backward()
    print(i, x.grad)
# 0 tensor(4.)
# 1 tensor(8.)            # 累加了！4+4

x.grad.zero_()            # 清零
y = x ** 2
y.backward()
print("after zero", x.grad)
# after zero tensor(4.)
```

训练循环标准顺序（务必记住）：

```python
# 伪代码
# optimizer.zero_grad()   # 或 param.grad.zero_()
# loss = model(x).mean()  # 前向
# loss.backward()         # 反传，写出 .grad
# optimizer.step()        # 用 .grad 更新参数
```

```python
w = torch.tensor(1.0, requires_grad=True)
for step in range(2):
    if w.grad is not None: #有东西才清零
        w.grad.zero_()
    loss = (w - 3) ** 2   # dloss/dw = 2(w-3)；w=1 时为 -4
    loss.backward()
    print("step", step, "grad", w.grad)
# step 0 grad tensor(-4.)
# step 1 grad tensor(-4.)   # 每步清零后一致
```
if w.grad is not None 是什么意思

w.grad 有两种状态：

None：还没做过 backward()，.grad 里什么都没有。
一个张量：已经做过 backward()，里面存着梯度。

is not None 就是"判断 .grad 里有没有东西"。


**易错点**：

1. **忘记 `zero_grad`** → 梯度越加越大，loss 乱飞。这是训练 №1 bug。
2. `optimizer.zero_grad()` 与手动 `p.grad.zero_()` 选一种即可；用优化器时用前者。
3. 刻意做「梯度累积」时（显存不够、等效大 batch）：连续多个 `backward` 再 `step`，中间**不要** `zero_grad`，但要注意 loss 要按累积步数平均。

**什么时候用**：每个 `optimizer.step()` 之前都必须保证本 step 的梯度是「新鲜的」——通常靠 step 开头 `zero_grad`。

---

## 4.4 `torch.no_grad()` 与 `torch.inference_mode()`

**是什么**：临时关闭梯度追踪，省内存、加快速度；验证 / 测试 / 推理时必用。

**怎么用**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x * 2
print(y.requires_grad)     # False

with torch.inference_mode():
    z = x * 2
print(z.requires_grad)     # False
```

也可用装饰器：`@torch.no_grad()` 包住 `evaluate()` 函数。

| API | 特点 |
|-----|------|
| `torch.no_grad()` | 常用；结果仍可在需要时再启用梯度相关操作 |
| `torch.inference_mode()` | 推理更快、限制更严；验证集优先推荐 |

**易错点**：

1. 验证循环忘了 `no_grad` → 显存暴涨、速度变慢。
2. 在 `no_grad` 里做了本该反传的运算 → 参数不更新却不报错，属于静默逻辑错误。
3. 仅 `model.eval()` **不会**关闭梯度；`eval` 只管 Dropout/BN 行为，梯度要靠 `no_grad`。

**什么时候用**：`validate` / `test` / 部署推理；以及「只用网络提特征、不训练」的代码块。

---

## 4.5 `detach()`：切断计算图

**是什么**：返回一个**共享数据、但不追踪历史**的新张量，后续运算不会反传到 `detach` 之前。

**怎么用**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3                 # requires_grad=True，有 grad_fn
z = y.detach()            # requires_grad=False
print(y.requires_grad, z.requires_grad)
# True False
```

**共享存储警告**：`detach()` 默认与原张量共享内存，原地改 `z` 可能改到 `y`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()
z.add_(1)
print(y, z)
# y 与 z 都变成 7.  （共享存储被改了）
```

需要独立副本时：

```python
z = y.detach().clone()    # 推荐：断图 + 拷贝
```

**易错点**：

1. 把 `detach` 当 `clone` 用，然后 `add_` → 污染原计算图里的中间量。
2. Actor-Critic、动量老师网络、把 loss 当系数用时常要 `detach`；忘了会「梯度串台」。
3. `x.numpy()` 要求张量不在 GPU 且不需梯度；常见写法 `x.detach().cpu().numpy()`。

**什么时候用**：

- 用网络输出当「常量目标」（如 target network）
- 记录日志 / 画图 / 转 NumPy
- 阻断某条支路的梯度

---

## 4.6 叶子节点、非标量 `backward`（最小必知）

### 非标量必须传入 `gradient`

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2                 # shape [2]，不是标量
try:
    y.backward()
except RuntimeError as e:
    print(type(e).__name__)
# RuntimeError: grad can be implicitly created only for scalar outputs

y = x * 2
y.backward(torch.ones_like(y))   # 等价于先对 y.sum() 再 backward
print(x.grad)
# tensor([2., 2.])
```

训练中更干净的做法：**永远对标量 loss 调用 `backward()`**，不要对向量 logits 直接反传。

### `retain_graph`（了解即可）

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print(x.grad)             # 4
y.backward()              # 第二次；梯度累加
print(x.grad)             # 8
```

一般训练每个 step 新建前向图，**不需要** `retain_graph=True`。多次对同一图反传（如某些高阶梯度）才用。

### 手动更新参数时

```python
# 演示：一轮「手写」梯度下降（正式训练请用优化器，见第 6 章）
torch.manual_seed(0)
x = torch.randn(4, 2)
y = torch.randn(4, 1)
W = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

pred = x @ W + b                    # [4, 1]
loss = ((pred - y) ** 2).mean()     # 标量
loss.backward()
print(loss.item())                  # 约 2.35（视随机而定）
print(W.grad.shape, b.grad.shape)   # [2, 1]  [1]

with torch.no_grad():               # 更新时不要建图
    W -= 0.1 * W.grad
    b -= 0.1 * b.grad
W.grad.zero_()
b.grad.zero_()
```

**易错点**：在 `W -= ...` 时若没包 `no_grad`，可能产生奇怪的图或报错；叶子参数更新一律在 `no_grad` 下进行（优化器内部已处理）。

---

## 4.7 训练一步心智图

```text
参数 θ (requires_grad=True)
        │
        ▼
   前向：loss = L(fθ(x), y)     ← 建图
        │
        ▼
   loss.backward()              ← 填 θ.grad
        │
        ▼
   optimizer.step()             ← θ ← θ - lr * grad
        │
        ▼
   optimizer.zero_grad()        ← 清掉 θ.grad（也可放在下一步开头）
```

验证阶段：

```text
model.eval()
with torch.no_grad():
    metric = evaluate(...)
```

---

## 本章小结

| 主题 | 记住 |
|------|------|
| 开梯度 | 浮点 + `requires_grad=True`（或 `nn.Parameter`） |
| 反传 | 对**标量** `loss.backward()`，读叶子 `.grad` |
| 清梯度 | 每步 `zero_grad`，否则累加 |
| 推理 | `model.eval()` + `torch.no_grad()`/`inference_mode()` |
| 断图 | `detach()`；要改数值用 `detach().clone()` |
| 非标量 | 需传 `gradient`，或先 `.sum()/.mean()` |

## 练习

1. 设 \(y = (x_1 + 2x_2)^2\)，`x=[1, 1]` 且 `requires_grad=True`，用 PyTorch 求出 \(\partial y/\partial x\)，并手算对照。
2. 故意去掉 `zero_grad`，跑 5 次 `backward`，观察 `.grad` 如何变化；写一句解释。
3. 写一个 `evaluate(model, loader)` 函数骨架，正确使用 `model.eval()` 与 `torch.no_grad()`。
4. 演示：`a = y.detach()` 后 `a.add_(1)` 是否影响 `y`；再对比 `y.detach().clone()`。
5. （综合）用纯 Tensor（不用 `nn.Module`）做 50 步线性回归：`y ≈ x @ W + b`，打印最后 loss；体会 `backward` / 更新 / `zero_grad` 闭环。
