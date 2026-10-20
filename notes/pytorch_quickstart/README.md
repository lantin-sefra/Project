# PyTorch 上手笔记

面向：有 Python / 线代基础、懂神经网络概念、未系统用过 PyTorch 的本科生。  
目标：读完并跑完代码后，能独立写出完整训练流程。

## 怎么读

1. 按章顺序学；每章末尾有「本章小结」和练习。
2. 代码优先：每个知识点按 **是什么 → 怎么用 → 常见坑 → 什么时候用**。
3. 理论补链：正文只给够写代码的直觉，深入理论通过链接跳转。
4. 所有示例基于 **PyTorch 2.x**；本仓库在 `2.1.2+rocm6.0` 上验证过。

## 设备约定（每章开头会再提一次）

本笔记在 **AMD ROCm** 环境编写：

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7600S 等需覆盖架构名
```

代码里统一写 `device="cuda"`。**NVIDIA CUDA 写法相同**——`torch.cuda.is_available()`、`.to("cuda")` 接口一致，只需安装对应 CUDA 版 PyTorch，无需改业务代码。

## 目录

| 章 | 文件 | 状态 |
|----|------|------|
| 0 | [ch00_how_to_use.md](./ch00_how_to_use.md) | ✅ |
| 1 | [ch01_environment.md](./ch01_environment.md) | ✅ |
| 2 | [ch02_tensor_basics.md](./ch02_tensor_basics.md) | ✅ |
| 3 | [ch03_math_ops.md](./ch03_math_ops.md) | ✅ |
| 4 | [ch04_autograd.md](./ch04_autograd.md) | ✅ |
| 5 | [ch05_0_roadmap.md](./ch05_0_roadmap.md)（路线与任务表）· [ch05_nn_module.md](./ch05_nn_module.md)（速查手册） | ✅ |
| 5.x | 子笔记：[5.1 Module 慢讲](./ch05_1_module_walkthrough.md) · [5.2 注册与参数](./ch05_2_registration_params.md) · [5.3 Linear](./ch05_3_Linear.md) · [5.4 shape 推导](./ch05_4_shape_drills.md) · [5.5 train/eval 与损失](./ch05_5_train_eval_loss.md) | ✅ |
| 6 | [ch06_optim_scheduler.md](./ch06_optim_scheduler.md) | ✅ |
| 7 | 数据加载 | 待写 |
| 8 | 完整训练流程 | 待写 |
| 9 | 综合实战：MNIST（MLP + 主线 CNN） | 待写 |
| 10 | 附录：报错速查与调试 | 待写 |


## 环境快速自检

```bash
source ~/venvs/dl/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望类似：`2.1.2+rocm6.0 True`
