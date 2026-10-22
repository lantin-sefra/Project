# learn_pytorch_and_numpy

按用途分成三个子目录，避免根目录堆成一锅粥。

| 目录 | 内容 |
|------|------|
| [`from_scratch/`](./from_scratch/) | 手写反传、线性/多项式回归、矩阵乘法梯度（个人简述里的「亮点」） |
| [`ex05/`](./ex05/) | 第 5 章 `nn.Module` 练习；`solutions/` 为参考解（卡 15 分钟再看） |
| [`scratch/`](./scratch/) | 早期试写与零散实验，不必按顺序看 |

## from_scratch/

| 文件 | 做什么 |
|------|--------|
| `two_layer_backprop.py` | NumPy 手写两层网络反传 + 方向导数检验 |
| `two_layer_backprop2.py` | 同一结构的 PyTorch autograd 对照版 |
| `matmul_backprop.py` | 矩阵乘法的反向传播（原「matmul的反向传播.py」） |
| `linreg.py` / `linreg_nn.py` | 线性回归：手写循环 → `nn.Module` |
| `polynomial_regression.py` | 多项式回归 |

```bash
cd from_scratch
python two_layer_backprop.py
python two_layer_backprop2.py
python linreg_nn.py
```

## ex05/

配套笔记：[ch05_0_roadmap.md](../notes/Math_and_pytorch_note/ch05_0_roadmap.md)

```bash
cd ex05
python ex1_module_basics.py   # … ex6_small_cnn.py
```

## scratch/

试写文件，文件名未统一整理。其中 `vectorize_speedup.py` 原名「矢量化加速.py」。
