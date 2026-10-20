# 第 1 章 · 环境与基础

> **设备提示**：AMD ROCm 用户请在运行前设置 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`（例如 RX 7600S）。代码里统一写 `device="cuda"`；**NVIDIA CUDA 写法相同**，只需安装对应 CUDA 版 wheel。

本章目标：装好 / 确认环境，会选 CPU 或 GPU，会固定随机种子——之后每一章都依赖这三件事。

---

## 1.1 安装与版本检查

**是什么**：确认 Python 能 `import torch`，并知道当前是 CPU 版、CUDA 版还是 ROCm 版。

**怎么用**：

```python
import torch

print(torch.__version__)
# 本机验证输出示例：
# 2.1.2+rocm6.0
```

版本字符串里的后缀含义：

| 后缀 | 含义 |
|------|------|
| 无 / `+cpu` | 仅 CPU |
| `+cu118` / `+cu121` 等 | NVIDIA CUDA |
| `+rocm6.0` 等 | AMD ROCm |

**易错点**：

1. **Python 小版本必须匹配 wheel**。例如 `cp310` 的包只能给 Python 3.10 用。
2. **NumPy 2.x 与较老的 PyTorch 2.1 不兼容**。若出现奇怪的 dtype / C-API 报错，可执行：`pip install "numpy<2"`。
3. 虚拟环境没激活时，装到了系统 Python，运行时却用另一个——永远先 `which python`。

**什么时候用**：每次换机器、新建 venv、怀疑装错版时，先打印 `__version__`。

📎 理论 / 官方文档：

- [PyTorch Get Started（选系统 / CUDA / 包管理器）](https://pytorch.org/get-started/locally/)
- [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)（装指定旧版时用）

---

## 1.2 GPU 是否可用

**是什么**：PyTorch 用 `torch.cuda` 这一套 API 管理加速设备（NVIDIA 与 ROCm 都走这个名字）。

**怎么用**：

```python
import torch

print(torch.cuda.is_available())   # True / False
print(torch.cuda.device_count())   # GPU 数量，常见为 1
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
# 本机（ROCm + RX 7600S）验证输出示例：
# True
# 1
# AMD Radeon Graphics
```

**易错点**：

1. ROCm 下 `is_available()` 为 `False`：常见原因是用户不在 `render`/`video` 组（需 `usermod` 后**重新登录**），或未设置 `HSA_OVERRIDE_GFX_VERSION`。
2. 有显卡但驱动 / 用户态库未装好时，也会是 `False`——先查系统，再查 PyTorch。
3. 不要用 `torch.version.cuda` 判断「能不能用 GPU」：ROCm 构建上该字段可能为 `None`，应以 `is_available()` 为准。

**什么时候用**：写训练脚本的第一行逻辑就应探测 device，避免写死 `"cuda"` 导致别人 CPU 机器直接崩。

📎 对比：NumPy 没有这套设备 API；CuPy / JAX 有各自的设备模型。

---

## 1.3 `device` 管理

**是什么**：每个 Tensor / 模块都住在某个设备上（`cpu` 或 `cuda:0`）。运算要求操作数在**同一设备**。

**怎么用**：

```python
import torch

# 推荐写法：自动回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 有 GPU 时：cuda
# 无 GPU 时：cpu

# 方式 A：先在 CPU 创建，再搬过去
x = torch.randn(2, 3)          # shape: [2, 3], device: cpu
print(x.device)                # cpu
y = x.to(device)                # shape: [2, 3], device: cuda:0（若可用）
print(y.device)

# 方式 B：创建时直接指定（少一次拷贝，推荐）
z = torch.ones(2, 3, device=device)
print(z.device, z.shape)
# 本机验证：cuda:0 torch.Size([2, 3])
```

把**模型**和**数据**都放到同一 device：

```python
# 伪代码示意（完整模型见后续章节）
# model = MyNet().to(device)
# for xb, yb in loader:
#     xb, yb = xb.to(device), yb.to(device)
#     pred = model(xb)
```

**易错点**：

1. **只把模型 `.to(cuda)`，忘了把 batch 数据也 `.to(device)`** → `Expected all tensors to be on the same device`。
2. `.cuda()` 在无 GPU 时会报错；`.to(device)` 更安全。
3. `.to(device)` 对 Tensor **默认返回新张量**（除非已在目标设备）；对 `nn.Module` 是**原地移动并返回自身**。习惯写成 `x = x.to(device)`、`model = model.to(device)`。

**什么时候用**：从第 8 章起的所有训练代码都以本节模板开头。

📎 官方：[Tensors · Accelerators](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#tensor-operations)（同页后半部分）

---

## 1.4 固定随机种子（可复现）

**是什么**：让随机初始化、数据 shuffle、`randn` 等在多次运行中尽量一致，方便对比实验与排错。
`set_seed()`做的是：把 Python / NumPy / PyTorch 的随机数生成器初始状态设成同一个起点。之后每次 `randn`、`rand` 等都会从这个状态往前走，整条随机序列就固定了。

**怎么用**：

```python
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 下面两行会降低一点性能，换绝对可复现时再开：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(42)
a = torch.randn(2, 2)
set_seed(42)
#中间又设一次种子，让a、b从同一起点开始，才能allclose
b = torch.randn(2, 2)
print(a)
print(b)
print("equal:", torch.allclose(a, b))
# 本机验证输出：
# tensor([[0.3367, 0.1288],
#         [0.2345, 0.2303]])
# tensor([[0.3367, 0.1288],
#         [0.2345, 0.2303]])
# equal: True
```

- 跨两次运行要可复现 → 每次启动脚本时设一次
- 同一次运行里连续采样 → 不要反复 set_seed，否则每次都会回到同一起点，后面的随机数会“重复开头那一段”

**易错点**：

1. 只设了 `torch.manual_seed`，忘了 `numpy` / `random`——`DataLoader` 的 shuffle 等仍可能不一致。
2. 多线程 `DataLoader(num_workers>0)` 时，完全 bitwise 可复现更麻烦；作业 / 小实验可先设 `num_workers=0`。
3. GPU 上仍可能有非确定性算子；科研复现需要时再开 `deterministic` 相关开关。

**什么时候用**：写论文实验、对比两个模型、调试「为什么这次 loss 不一样」时。日常探索可以不固定。

📎 理论：随机初始化与训练动态见 [d2l · 数值稳定性与模型初始化](https://zh.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)

---

## 1.5 最小「CPU/GPU 算一下」冒烟测试

确认整条链路：创建 → 上卡 → 运算 → 看 shape。

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a: [4, 8], b: [8, 3]  ->  c: [4, 3]
a = torch.randn(4, 8, device=device)
b = torch.randn(8, 3, device=device)
c = a @ b

print(c.shape, c.device)
# 本机 GPU 验证：torch.Size([4, 3]) cuda:0
```

若这里报错，先不要进第 2 章——把环境问题清掉再继续。

---

## 1.6 （可选）本机 ROCm 安装备忘

你当前机器已验证可用，仅作存档。新机器可参考：

```bash
sudo apt install -y python3.10-venv python3-pip
sudo usermod -aG render,video $USER   # 然后重新登录

python3 -m venv ~/venvs/dl
source ~/venvs/dl/bin/activate
pip install -U pip
pip install ~/Downloads/torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl \
            ~/Downloads/torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl
pip install "numpy<2" matplotlib pandas tqdm

echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
```

NVIDIA 用户改为从 [Get Started](https://pytorch.org/get-started/locally/) 复制对应 `pip install torch` 命令即可，**业务代码不用改**。

---

## 本章小结

| 要点 | 记住 |
|------|------|
| 版本 | `torch.__version__`，注意 `+rocm` / `+cu` / CPU |
| 探测 GPU | `torch.cuda.is_available()` |
| 设备 | `device = torch.device(...)`，模型和数据都 `.to(device)` |
| 可复现 | `set_seed`；多库一起设 |
| ROCm | `HSA_OVERRIDE_GFX_VERSION`；API 仍叫 `cuda` |

## 练习

1. 在本机运行 1.1–1.5 全部代码，把输出贴进自己的学习日志。
2. 故意制造一次 device 错误：模型在 GPU、输入在 CPU，观察完整报错信息（认识它，以后好查）。
3. 将 `set_seed(42)` 改成 `set_seed(0)`，确认 `randn` 结果变化；再改回 42，确认与笔记中的张量一致（CPU 上应一致）。
4. （思考）为什么 ROCm 版 PyTorch 仍使用 `torch.cuda` 这个名字？对写跨平台脚本有什么好处？
5. 阅读 [PyTorch Get Started](https://pytorch.org/get-started/locally/)，写出「若你只有 NVIDIA 笔记本，应选择的三条安装命令关键词」（系统 / pip / CUDA 版本）。
