# 第 0 章 · 怎么用这份笔记

> **设备提示**：AMD ROCm 用户请在运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。代码里写 `device="cuda"`；NVIDIA CUDA 写法相同。

---

## 0.1 这份笔记解决什么问题

你已经知道「前向 → 损失 → 反向 → 更新参数」在干什么，但还没把这条链路稳定地写成能跑的 PyTorch 代码。

读完并动手后，你应能：

1. 用 Tensor 表达数据和参数，清楚每次运算的 **shape**
2. 写出 `nn.Module`、用 `DataLoader` 喂数据、用优化器更新
3. 独立完成 **训练 / 验证 / 保存 / 加载** 闭环（第 9 章项目）

---

## 0.2 推荐学习路径

| 节奏 | 建议 |
|------|------|
| 高强度（国庆冲刺） | 每天 1 章 + 当晚做练习；第 9 章留一整天 |
| 稳妥（2 周） | 工作日第 1–8 章，周末第 9 章 + 附录 |
| 卡壳时 | 先保证「最小可运行示例」能跑，再点理论链接 |

**不要**先通读西瓜书 / mmlbook 再写代码。正确顺序是：

> 写到哪 → 卡在哪 → 查哪一节理论 → 回到代码

---

## 0.3 代码约定（全书统一）

### shape 注释

凡涉及张量运算，注释里写清输入 / 输出形状：

```python
# x: [B, 784]  ->  logits: [B, 10]
logits = model(x)
```

`B` = batch size，后面章节会反复出现。

### device 一行搞定

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

### 可复现：固定种子

训练对比实验时务必固定；探索调参时可关掉。写法见第 1 章。

### 不用已废弃 API

- ❌ `Variable`（PyTorch 0.4 起并入 Tensor）
- ✅ 直接对 `Tensor` 设 `requires_grad=True`，或用 `nn.Parameter`

---

## 0.4 与 NumPy 的心智对照

| 概念 | NumPy | PyTorch |
|------|-------|---------|
| 多维数组 | `ndarray` | `Tensor` |
| 创建 | `np.zeros` | `torch.zeros` |
| 矩阵乘 | `A @ B` / `np.dot` | `A @ B` / `torch.matmul` |
| 形状 | `.shape` | `.shape`（相同） |
| 求导 | 手写或别的库 | **内置 autograd** |
| GPU | 基本没有 | `.to("cuda")` |

多数 NumPy 写法可以「几乎逐字」改成 PyTorch；真正新的只有三块：**自动求导、`nn.Module`、DataLoader**。

理论补充：

- [NumPy 官方绝对入门](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [PyTorch 官方 Tutorials · Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)

---

## 0.5 理论补链怎么用

正文里会用「📎 理论」标出链接，例如：

- 梯度下降 / 反向传播：李沐《动手学深度学习》[线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)、[多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- 矩阵微积分：[The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)（查阅用）
- 更系统的数学：[Mathematics for Machine Learning (mmlbook)](https://mml-book.github.io/) 第 5 章 Vector Calculus

**第一遍以跑通代码为准**；链接留给你「想弄懂为什么」时点开。

---

## 0.6 文件与运行方式

```text
notes/pytorch_quickstart/
├── README.md
├── ch00_how_to_use.md      ← 你在这里
├── ch01_environment.md
└── ...
```

每章示例可复制到 `python` 交互环境或 `.py` 文件运行。建议：

```bash
source ~/venvs/dl/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd /home/rwr/Project/notes/pytorch_quickstart
# 把代码贴进临时脚本或 jupyter
```

---

## 本章小结

- 目标是写出完整训练流程，不是背 API。
- 统一约定：shape 注释、`device`、种子、不用 Variable。
- PyTorch ≈ NumPy + 自动求导 + 模块化神经网络 + 数据管道。

## 练习

1. 用一句话向同学解释：为什么学 PyTorch 前不必把西瓜书读完。
2. 打开 [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)，浏览目录，标出你最陌生的 2 节。
3. 确认本机 `import torch` 成功（不会的话先做第 1 章）。
