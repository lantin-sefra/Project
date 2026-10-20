# 第 6 章 · 优化器与学习率调度

> **设备提示**：AMD ROCm 用户运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码写 `device="cuda"`；NVIDIA CUDA 写法相同。本章示例 CPU 即可。

第 4 章解决了「梯度从哪来」，本章解决「参数怎么更新」以及「学习率怎么随训练变化」。  
目标：熟练写出 `zero_grad → backward → step`，会选 SGD/Adam/AdamW，会挂上常用 `lr_scheduler`。

📎 理论：

- [d2l · 优化算法](https://zh.d2l.ai/chapter_optimization/index.html)（SGD、动量、Adam）
- [d2l · 学习率调度](https://zh.d2l.ai/chapter_optimization/lr-scheduler.html)
- Adam 原文：[Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- AdamW 原文：[Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101)
- OneCycle：[Smith & Topin, 2017](https://arxiv.org/abs/1708.07120)

---

## 6.1 优化器共性接口

**是什么**：优化器持有参数引用，根据 `.grad` 按规则更新权重。

**标准一步**（务必背熟）：

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
model = nn.Linear(4, 1)
opt = optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(8, 4)
y = torch.randn(8, 1)

for step in range(3):
    opt.zero_grad()                         # 1. 清梯度
    loss = ((model(x) - y) ** 2).mean()     # 2. 前向
    loss.backward()                         # 3. 反传，写入 .grad
    opt.step()                              # 4. 用 .grad 更新参数
    print(step, f"{loss.item():.4f}")
# 0 3.1121
# 1 1.6428
# 2 1.0707
```

读当前学习率：

```python
print(opt.param_groups[0]["lr"])
```

**易错点**：

1. 顺序写反：`step` 后再 `backward`，或忘记 `zero_grad`（第 4 章）。
2. `opt = Adam(model.parameters())` 之后又改了模型结构 / 新加层 → 新参数不在优化器里，不会更新。
3. 手动改 `param.requires_grad` 后，优化器仍可能留着旧引用；冻结层应在**创建优化器之前**设好，或使用 param group。

**什么时候用**：有 `nn.Module` 且需要训练时，几乎总是用优化器，而不是手写 `w -= lr * w.grad`。

---

## 6.2 `SGD` / `Adam` / `AdamW`

**是什么**：三种最常用的一阶优化器；区别在「怎么用历史梯度」和「权重衰减怎么加」。

### SGD（可加动量）

```python
opt = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,          # 常用；0 则是正统 SGD
    weight_decay=1e-4,     # L2 正则（耦合在更新里）
)
```

### Adam

```python
opt = optim.Adam(model.parameters(), lr=1e-3)   # 深度学习默认起点
print(opt.param_groups[0]["lr"])
# 0.001
```

### AdamW（更推荐的默认）

```python
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
print(opt.param_groups[0]["weight_decay"])
# 0.01
```

Adam 里的 `weight_decay` 实现与「解耦的权重衰减」不完全等价；**Transformer / 现代训练脚本默认 AdamW**。你做 CNN / MLP 作业，AdamW 也通常是稳妥选择。

### 怎么选（实用版）

| 场景 | 建议 |
|------|------|
| 快速出基线 / 大部分深度学习 | **AdamW**，`lr≈1e-3`，再按验证集调 |
| CNN 经典配方、要抠最后一点精度 | SGD + momentum=0.9，配合学习率衰减 |
| 微调大模型、BERT 类 | AdamW + 较小 lr（如 `2e-5`）+ 往往配合 warmup |
| 损失剧烈震荡 | 先把 lr 降 10 倍，再换优化器 |

### 参数组：不同层不同 lr

```python
backbone = nn.Linear(10, 10)
head = nn.Linear(10, 2)
opt = optim.Adam([
    {"params": backbone.parameters(), "lr": 1e-4},  # 预训练骨干小学习率
    {"params": head.parameters(), "lr": 1e-3},       # 新头大学习率
])
print([g["lr"] for g in opt.param_groups])
# [0.0001, 0.001]
```

**易错点**：

1. SGD 的 lr 常比 Adam 大一个数量级（如 0.1 vs 1e-3），照抄会训崩。
2. `weight_decay` 不是 0 时，等价于给 loss 加了正则；和「自己在 loss 里加 L2」不要重复叠太狠。
3. 只把部分 `parameters()` 传进优化器 → 其余层永远不更新（有时是故意冻结，有时是 bug）。

📎 理论：动量与自适应学习率直观对比见 [d2l · Adam](https://zh.d2l.ai/chapter_optimization/adam.html)

---

## 6.3 学习率调度 `lr_scheduler`

**是什么**：按 epoch 或按 batch 自动改 `optimizer.param_groups[..]["lr"]`。

**通用模式**：

```text
每个 epoch（或每个 batch）:
    训练若干 step（内部：zero_grad → backward → optimizer.step）
    scheduler.step(...)     # 多数按 epoch；OneCycle 常按 batch
```

### ① `StepLR`：阶梯式衰减

每隔 `step_size` 个 epoch，lr 乘以 `gamma`。

```python
model = nn.Linear(4, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

lrs = []
for epoch in range(6):
    lrs.append(optimizer.param_groups[0]["lr"])
    # ... 正常训练 ...
    optimizer.zero_grad()
    loss = model(torch.randn(4, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()          # 每个 epoch 末调用一次

print(lrs)
# [0.1, 0.1, 0.01, 0.01, 0.001, 0.001]
```

### ② `CosineAnnealingLR`：余弦退火

从初始 lr 平滑降到 `eta_min`，一个周期长度为 `T_max` 个 epoch。

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=5, eta_min=0.01
)
lrs = []
for epoch in range(6):
    lrs.append(round(optimizer.param_groups[0]["lr"], 4))
    optimizer.step()          # 演示用；真实训练里在 batch 循环内
    scheduler.step()
print(lrs)
# [0.1, 0.0914, 0.0689, 0.0411, 0.0186, 0.01]
```

现代训练（含不少扩散 / ViT 配方）很爱余弦衰减。

### ③ `OneCycleLR`：单周期（常按 **batch** 步进）

先升后降，常能用更短的训练快速收敛；**必须**知道总 step 数。

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 这里的 lr 会被 OneCycle 覆盖策略接管
steps_per_epoch = 4
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
)

lrs = []
for epoch in range(3):
    for batch in range(steps_per_epoch):
        lrs.append(round(optimizer.param_groups[0]["lr"], 4))
        optimizer.step()
        scheduler.step()      # 注意：每个 batch 调一次
print(lrs[0], lrs[len(lrs) // 2], lrs[-1], len(lrs))
# 0.004 0.0647 0.0  12
```

### ④ `ReduceLROnPlateau`：监控指标再降（可选了解）

验证集长期不降才衰减；`step` 要传入指标。

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1
)
for val_loss in [1.0, 0.9, 0.91, 0.92, 0.5]:
    scheduler.step(val_loss)   # 传入监控值
    print(val_loss, optimizer.param_groups[0]["lr"])
# 1.0  0.1
# 0.9  0.1
# 0.91 0.1
# 0.92 0.05      # patience 用尽，lr *= 0.5
# 0.5  0.05
```

### 调用时机对照

| Scheduler | 通常何时 `scheduler.step` |
|-----------|---------------------------|
| `StepLR` / `CosineAnnealingLR` / `MultiStepLR` | **每个 epoch 结束**一次 |
| `OneCycleLR` / `CyclicLR` | **每个 batch**（`optimizer.step` 之后） |
| `ReduceLROnPlateau` | 每个 epoch 验证后，`step(val_metric)` |

查看调度后的 lr：

```python
print(scheduler.get_last_lr())   # list，与 param_groups 对应
```

**易错点**：

1. **OneCycle 却按 epoch 才 step 一次** → 衰减节奏全错。
2. `scheduler.step()` 写在 `optimizer.step()` **之前**（旧版警告过）；PyTorch 1.1+ 推荐先 `optimizer.step()` 再 `scheduler.step()`。
3. 断点续训时既要恢复 `optimizer.state_dict()`，也要恢复 `scheduler.state_dict()`（第 8 章）。
4. 打印 `param_groups[0]["lr"]` 若在 `step` 前后搞混，会对不上曲线。

**什么时候用**：

- 作业 / 小数据集：`StepLR` 或固定 lr 就够
- 想稳一点训满较长 epoch：`CosineAnnealingLR`
- 想短周期冲准确率：`OneCycleLR`
- 懒得设衰减点、看验证集说话：`ReduceLROnPlateau`

---

## 6.4 最小完整模板（可直接粘到训练脚本）

```python
model = nn.Linear(784, 10)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:          # 第 7 章
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()                     # 按 epoch 调度
    print(epoch, scheduler.get_last_lr())
```

梯度裁剪（RNN / 大模型常加，了解即可）：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 本章小结

| 主题 | 记住 |
|------|------|
| 四步 | `zero_grad` → 前向 → `backward` → `step` |
| 默认优化器 | 作业/深度学习优先 **AdamW**；经典 CNN 可试 SGD+momentum |
| 学习率 | Adam 系常从 `1e-3` 起；SGD 常更大 |
| 调度 | Step/Cosine 按 epoch；OneCycle 按 batch；Plateau 传指标 |
| 顺序 | 先 `optimizer.step()`，再 `scheduler.step()` |

## 练习

1. 用同一 `nn.Linear` 回归任务，分别用 SGD（lr=0.1）和 AdamW（lr=1e-2）训 50 步，画两条 loss 曲线（或打印每 10 步）。
2. 挂上 `StepLR(step_size=10, gamma=0.5)`，跑 30 个 epoch，记录每个 epoch 的 lr 列表，确认阶梯位置。
3. 把调度器换成 `CosineAnnealingLR(T_max=30, eta_min=1e-4)`，对比最终 loss 与 StepLR。
4. 故意对 `OneCycleLR` 每个 epoch 只 `scheduler.step()` 一次，观察 lr 变化是否异常，并写一句解释。
5. （综合）为「骨干 + 分类头」建两个 param group（lr 相差 10 倍），打印 `param_groups`，确认设置生效。
