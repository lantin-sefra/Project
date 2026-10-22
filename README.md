# Project

一个小仓库，我目前在学习 PyTorch / 深度学习 / 强化学习。这里面保存了我的实验和笔记。我会经常更新的。
我的环境是 Ubuntu 22.04 · Python 3.10 · PyTorch `2.1.2+rocm6.0`  因为是AMD显卡，所以不能使用cuda，不过通过覆写设备的哈希值，我还是能够实现显卡加速。小网络 RL 优先 CPU；用 GPU 前设 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`。

给后续 AI agent 的上下文交接：[handoff.md](handoff.md)（目标导师、学习阶梯、已否决的建议、环境约束）。



目前正在学：

| 入口                                                                                     | 说明                |
| -------------------------------------------------------------------------------------- | ----------------- |
| [RL 第 1 章 · 表格型 Q-learning（FrozenLake）](notes/RL_note/ch01_q_learning_frozenlake.md) | **当前主线**；脚本见下 |
| [矩阵微分：用 Frobenius 内积读梯度](notes/Math_and_pytorch_note/matrix_calculus_frobenius.md)     | 矩阵梯度推导主线          |
| [公式速查](notes/Math_and_pytorch_note/matrix_calculus_cheatsheet.md)                      | 上面这篇的公式清单         |
| [softmax ↔ 极大似然](notes/Math_and_pytorch_note/softmax_log-likelihood.md)                | 交叉熵与 MLE 的对应      |
| [手写两层网络反向传播](learn_pytorch_and_numpy/from_scratch/two_layer_backprop.py)               | NumPy 实现 + 方向导数检验 |
| [线性回归 →](learn_pytorch_and_numpy/from_scratch/linreg.py) `nn.Module`                   | 对照 `linreg_nn.py` |
| [第 5 章练习（](learn_pytorch_and_numpy/ex05/)`nn.Module`[）](learn_pytorch_and_numpy/ex05/) | 带 assert 的 6 个小练习 |

完整笔记目录见 [notes/Math_and_pytorch_note/README.md](notes/Math_and_pytorch_note/README.md) · [notes/RL_note/README.md](notes/RL_note/README.md)。

目录结构

```
learn_pytorch_and_numpy/
  from_scratch/   # 手写反传、线性回归、多项式回归等
  ex05/           # D2L / nn.Module 练习（solutions/ 为参考解）
  scratch/        # 早期试写、零散实验
notes/
  Math_and_pytorch_note/   # PyTorch 上手笔记 + 矩阵微积分
  RL_note/                 # 强化学习笔记（Q-learning → DQN → SAC …）
experiments/
  rl/                      # RL 自写实验脚本
Numpy-100/                 # NumPy 100 题（嵌套仓库）
```


可以这样运行
```bash
cd learn_pytorch_and_numpy/from_scratch
python two_layer_backprop.py
python linreg_nn.py

cd ../ex05
python ex1_module_basics.py

# 强化学习（推荐用 /home/rwr/venvs/dl/bin/python）
cd ../../experiments/rl
python q_learning_frozenlake.py          # 全部实验，约 2 分钟，CPU
python q_learning_frozenlake.py det      # 只跑确定性 FrozenLake
python q_learning_frozenlake.py slip     # 滑冰版 + 3 seed
```

进度

- [x] 矩阵微分（Frobenius）与手写反向传播
- [x] D2L 第 1–5 章相关实验（线性模型 / MLP / `nn.Module`）
- [x] 小型 CNN 练习
- [x] 表格型 Q-learning @ FrozenLake（笔记 + 自写脚本已跑通；待个人验收）
- [ ] 强化学习（DQN → SAC）← **下一站：DQN @ CartPole**
