# 5.4 shape 推导：写 CNN 的基本功

> 写网络时最常见的报错是 shape 对不上。本篇给你一套**固定流程**，把「靠猜」变成「靠算」。
> 前置：[5.3 Linear](./ch05_3_Linear.md)。配套练习：`ex05/ex3_shapes.py`

---
$$ \large x = torch.tensor([B,C_{in},H,W])$$

## 1. 每层只需要记一句话

| 层 | 规则 | 例 |
|---|---|---|
| `Linear(in, out)` | 只改**最后一维**：`in → out`，前面不动 | `[B,20] → [B,30]` |
| `Conv2d(Cin, Cout, k, s, p)` | `C: Cin → Cout`；`H,W` 按公式变 | `[B,3,32,32] → [B,16,?,?]` |
| `MaxPool2d(k)` | `C` 不变；`H,W` 变成原来的 `1/k`（向下取整） | `[B,16,28,28] → [B,16,14,14]` |
| `ReLU` / `BatchNorm2d` / `Dropout` | **shape 完全不变** | |
| `Flatten()` | 保留 `B`，其余乘起来 | `[B,64,7,7] → [B,3136]` |

### 卷积/池化的空间公式

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

池化同理（池化默认 `stride = kernel_size`，`p = 0`）：`MaxPool2d(2)` 即 `k=2,s=2`，`(H-2)/2+1 = H/2`。

**三个必背的常见组合**：

| 配置 | 效果 |
|---|---|
| `k=3, s=1, p=1` | `H,W` **不变**（最常用的「保形卷积」） |
| `k=2` 的 `MaxPool` | `H,W` **减半** |
| `k=5, s=2, p=0`（32 输入） | `(32-5)/2+1 = 14.5 → 14` |

---

## 2. 通道只由你「指定」，空间由「公式」算

很多人把这两件事混在一起。分开想：

- **通道数 C**：就是你写在 `Conv2d(Cin, Cout, ...)` 里的 `Cout`，**随便选**，唯一约束是下一层的 `Cin` 必须等于上一层的 `Cout`。
- **空间 H,W**：由 `k, s, p` 和池化决定，**要算**。

---

## 3. 完整追踪示例

输入 `[8, 3, 32, 32]`，网络：

```python
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),    # 1
    nn.ReLU(),                         # 2
    nn.MaxPool2d(2),                   # 3
    nn.Conv2d(32, 64, 3, padding=1),   # 4
    nn.ReLU(),                         # 5
    nn.MaxPool2d(2),                   # 6
)
```

逐层追踪（已用代码验证）：

| 层 | 输出 shape | 原因 |
|---|---|---|
| 输入 | `[8, 3, 32, 32]` | |
| 1 Conv(3→32, k3 p1) | `[8, 32, 32, 32]` | 通道变；`k=3,p=1` 保形 |
| 2 ReLU | `[8, 32, 32, 32]` | 不变 |
| 3 MaxPool(2) | `[8, 32, 16, 16]` | 减半 |
| 4 Conv(32→64, k3 p1) | `[8, 64, 16, 16]` | |
| 5 ReLU | `[8, 64, 16, 16]` | |
| 6 MaxPool(2) | `[8, 64, 8, 8]` | |

然后 `Flatten` → `[8, 64*8*8] = [8, 4096]`，下一层 `Linear` 的 `in_features` 就是 **4096**。

### 让代码替你算：逐层打印

不要硬算整网，用切片逐层看：

```python
x = torch.randn(8, 3, 32, 32)
for i in range(1, len(net) + 1):
    print(i, net[:i](x).shape)
```

`net[:i]` 取前 `i` 层。这个技巧在调试「某一层突然对不上」时极其好用。

---

## 4. `Linear` 的 `in_features` 怎么确定

接在卷积后面的 `Linear`，`in_features` = `C × H × W`（Flatten 之后）。三种办法，按推荐顺序：

1. **先 `print` 再填**：搭完卷积部分，喂一个假输入看 shape，再算 `C*H*W`。
2. **在 `forward` 里先 `print(x.shape)`**：跑一次后再删。
3. 用 `nn.AdaptiveAvgPool2d((1,1))` 把空间压成 `1×1`，则 `in_features = C`，不再依赖输入尺寸。

```python
nn.Sequential(net, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 10))
```

---

## 5. 报错对照

| 报错 | 原因 | 怎么查 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (32x3136 and 1568x128)` | Flatten 后的特征数 ≠ `Linear.in_features` | 把 `3136` 反推：`64*7*7`；你填的 `1568 = 32*7*7`，通道数填错了 |
| `Given groups=1, weight of size [8, 3, 3, 3], expected input[1, 1, 28, 28] to have 3 channels, but got 1 channels instead` | `Conv2d(in_channels)` 与输入通道数不一致 | 看 `expected input[...]` 里第 2 维：MNIST 是 1 通道，CIFAR 是 3 |
| 同上，但 `expected input[1, 32, 32, 3] to have 3 channels, but got 32 channels` | 通道维位置不对（NHWC 当 NCHW 喂了） | 报错里 `got 32 channels` 说明 32 被当成了通道；PyTorch 是 NCHW，用 `x.permute(0,3,1,2)` |
| `Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3)` | 特征图太小，卷积核比图还大 | 少池化几次，或减小核 |

---

## 小结

- 先分清：通道是你选的，空间是公式算的。
- 记住保形卷积 `k3 s1 p1`，池化 `k2` 减半。
- 不确定时：`net[:i](x).shape` 逐层打印。
- Flatten 后的数 = `C×H×W` = 下一层 `Linear` 的 `in_features`。

## 任务

→ 做 `ex05/ex3_shapes.py`（每题先笔算预测，再运行）。通过后进入 [5.5 train/eval 与损失](./ch05_5_train_eval_loss.md)。
