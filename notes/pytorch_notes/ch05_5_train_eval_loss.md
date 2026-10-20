# 5.5 `train()/eval()`、损失函数与完整训练循环

> 目标：知道为什么验证前必须 `eval()`；分类损失一次选对；能用 `nn` 写出完整训练循环。
> 前置：[5.1](./ch05_1_module_walkthrough.md)。配套练习：`ex05/ex4_train_eval.py`、`ex05/ex5_train_mlp.py`

---

## 1. 训练循环：把 `linreg_nn.py` 画成一张图

```python
for epoch in range(num_epochs):
    model.train()                       # ① 进入训练模式
    for X, y in loader:
        logits = model(X)               # ② 前向
        loss = loss_fn(logits, y)       # ③ 算损失（标量）
        optimizer.zero_grad()           # ④ 清掉上一步的梯度（因为 backward 是累加）
        loss.backward()                 # ⑤ 反向：每个参数的 .grad 被填好
        optimizer.step()                # ⑥ 更新：param -= lr * grad
    model.eval()                        # ⑦ 验证前切推理模式
    with torch.no_grad():               # ⑧ 验证不需要梯度
        ...
```

顺序口诀：**前向 → 损失 → 清零 → 反向 → 更新**。④ 放在 ③ 之后、⑤ 之前都行，但必须在 ⑤ 之前、且每步一次。

| 步骤 | 如果漏掉 |
|---|---|
| `zero_grad` | 梯度累加，越训越乱（见 5.1 实验 C） |
| `backward` | `.grad` 是 `None`，`step()` 什么都不做 |
| `step` | 参数永远不变，loss 不动 |
| `eval()` | Dropout/BN 仍按训练行为，验证指标抖动 |
| `no_grad()` | 能跑，但白占显存、变慢 |

---

## 2. 为什么需要 `train()/eval()`：两个会「变脸」的层

只有两类层在训练和推理时**行为不同**：`Dropout`、`BatchNorm`。其他层（Linear、Conv、ReLU）完全无所谓。

### 2.1 Dropout

```python
drop = nn.Dropout(0.5)
x = torch.ones(1, 10000)
drop.train()
y = drop(x)
print((y == 0).float().mean().item())   # 预测？
print(y.max().item())                   # 预测？
drop.eval()
print(torch.equal(drop(x), x))          # 预测？
```

<details><summary>答案</summary>

约 `0.5`；`2.0`；`True`。

- 训练：随机丢掉一半，存活的元素乘 `1/(1-p) = 2`，这样**期望值保持不变**（inverted dropout）。
- 推理：什么都不做，原样输出。
</details>

### 2.2 BatchNorm

```python
bn = nn.BatchNorm1d(3)
data = torch.randn(64, 3) * 5 + 2
print(bn.running_mean, bn.num_batches_tracked)   # 预测：全 0，0
bn.train(); bn(data)
print(bn.running_mean, bn.num_batches_tracked)   # 预测：变了，1
bn.eval(); bn(data)                              # eval 模式下前向，running_mean 会变吗？
```

<details><summary>答案</summary>

初始 `tensor([0.,0.,0.])`、`0`；一次训练前向后 `running_mean` 被更新（向 batch 均值靠近一点）、`num_batches_tracked=1`；`eval()` 下前向**不更新**。

- **训练**：用**当前 batch** 的均值和方差做归一化，并顺手更新 `running_mean/var`（滑动平均）。
- **推理**：用存下来的 `running_mean/var`，不依赖当前 batch，所以单样本也能推理。
</details>

**一个实验看清差别**：同一批数据，train 和 eval 下归一化后的均值不一样：

```python
bn.train(); print(bn(data).mean(0))   # ≈ 0（用本 batch 的统计）
bn.eval();  print(bn(data).mean(0))   # 不是 0（用的是 running 统计，还没收敛）
```

### 2.3 一个必养成的习惯

```python
model.train()   # 每个 epoch 训练前
model.eval()    # 每次验证/测试/推理前
```

忘了 `model.eval()` 是最常见的「验证指标莫名其妙」的原因。

---

## 3. 分类损失：一次选对

先问两个问题：**输出怎么解读？标签长什么样？**

| 任务 | 模型输出（最后一层，不加激活） | 标签 | 损失 |
|---|---|---|---|
| 多分类（单选，C 类） | `[B, C]` logits | `[B]`，**`long`**，值∈[0,C-1] | `CrossEntropyLoss` |
| 二分类 / 多标签 | `[B]` 或 `[B,1]` logits | 同 shape，**`float`**，0/1 | `BCEWithLogitsLoss` |
| 回归 | `[B, 1]` | 同 shape，`float` | `MSELoss` |

### 铁律：输出 logits，不要手动 softmax / sigmoid

`CrossEntropyLoss` 内部已经做了 `log_softmax`，`BCEWithLogitsLoss` 内部已经做了 sigmoid（并用了数值更稳定的写法）。

```python
logits = torch.randn(4, 10)
t = torch.tensor([1, 3, 0, 7])
F.cross_entropy(logits, t)                                   # 正确
F.nll_loss(F.log_softmax(logits, dim=1), t)                  # 等价写法，数值相同
F.cross_entropy(F.softmax(logits, dim=1), t)                 # ❌ softmax 做了两次
```

同一份 `logits` 下，前两行数值相同（例如都是 `2.2948`），第三行不同（例如 `2.2754`；具体数字随随机数而变，但「前两个相等、第三个不等」是固定的）。**错误接法不会报错，只是 loss 数值不对、梯度被压扁**，模型照样能训，只是训得差——所以必须靠记忆而不是靠报错。

### 标签类型报错对照

| 现象 | 原因 |
|---|---|
| `expected scalar type Long but found Float` | `CrossEntropyLoss` 的标签必须是 `long`：`y.long()` |
| `result type Float can't be cast to the desired output type Long` | `BCEWithLogitsLoss` 的标签给成了 `long`，要 `float`：`y.float()` |
| `Target size (torch.Size([4])) must be the same as input size (torch.Size([4, 1]))` | `[B]` 和 `[B,1]` 混用：`y.view(-1, 1)` 或 `logits.squeeze(1)` |
| `0D or 1D target tensor expected, multi-target not supported` | `CrossEntropyLoss` 的标签多了一维（`[B,1]`）：用 `y.squeeze(1)` 变成 `[B]` |

### 预测时才用 softmax

```python
probs = torch.softmax(logits, dim=1)       # 要概率时才用
pred = logits.argmax(dim=1)                # 只要类别：argmax(logits) 与 argmax(probs) 相同，无需 softmax
```

---

## 4. 把它们串起来：XOR 小实验（ex5）

数据：`y = 1 if x0*x1 > 0 else 0`。这是个**线性不可分**问题。

| 模型 | 预期准确率 |
|---|---|
| `Linear(2,16) → Linear(16,2)`（无激活） | ≈ 0.5（和瞎猜差不多） |
| `Linear(2,16) → ReLU → Linear(16,2)` | > 0.9 |

这个对比说明了 `nn` 里最重要的一件事：**没有非线性激活，叠多少层 Linear 都等价于一层 Linear**（\(W_2(W_1x+b_1)+b_2 = (W_2W_1)x + \dots\)）。

---

## 小结

- 训练循环：前向 → 损失 → `zero_grad` → `backward` → `step`；训练前 `train()`，验证前 `eval()` + `no_grad()`。
- Dropout / BN 在 train/eval 下行为不同，其余层无所谓。
- 多分类：logits + `long` 标签 + `CrossEntropyLoss`，不要自己 softmax。
- 没有非线性激活，深度没有意义。

## 任务

→ 做 `ex05/ex4_train_eval.py`，再做 `ex05/ex5_train_mlp.py`。通过后做 `ex05/ex6_small_cnn.py` 作为本章综合。
