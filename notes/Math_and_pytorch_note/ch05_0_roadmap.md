# 第 5 章 · `torch.nn` 学习路线与任务表

> 这章原来的 [`ch05_nn_module.md`](./ch05_nn_module.md) 是**速查手册**（API 一览）。
> 想「真正会用」，按本页的顺序走：**慢讲子笔记 → 做练习 → 过验收**。手册留着查。

---

## 为什么要这样组织

| 问题 | 解法 |
|---|---|
| 一次摊开 Module、容器、Conv、BN、LSTM… 太快 | 拆成 4 个主题，每个只解决一个「会用」的障碍 |
| 代码和结果都给好了，只是在「读」 | 子笔记里每个实验**先预测再看答案** |
| 不知道自己会没会 | 每个练习是带 `assert` 的骨架，通过 = 会了 |
| 所有层同等重要 | 分级：必会 / 会查 / 了解 |

---

## 内容分级

| 级别 | 内容 | 要求 |
|---|---|---|
| **必会**（第 9 章直接用） | `Module` 三件事、`Linear`、`Conv2d`/`MaxPool2d`/`Flatten`、`ReLU`、`Sequential`/`ModuleList`、`CrossEntropyLoss`、`train/eval`、`parameters`/`state_dict` | 不看笔记能写出来 |
| **会查** | `BatchNorm`、`Dropout`、`LayerNorm`、`BCEWithLogitsLoss`、`MSELoss`、`F.xxx` 与 `nn.xxx` 的区别 | 知道在哪一节，知道各自的坑 |
| **了解** | `Embedding`、`RNN/LSTM`、`ModuleDict`、`register_buffer` | 用到时再读手册对应节，**现在不用练** |

---

## 任务表

每个任务的节奏：**读 15–25 min（边读边做预测） → 做练习 20–40 min**。练习文件在 `learn_pytorch_and_numpy/ex05/`。

| # | 读什么 | 做什么 | 验收（全部满足才算过） | 预计 |
|---|---|---|---|---|
| 1 | [5.1 Module 慢讲](./ch05_1_module_walkthrough.md) | `ex1_module_basics.py` | 脚本打印 `ex1 OK`；**合上笔记**能默写 `__init__ + forward` 骨架 | 1 h |
| 2 | [5.2 注册机制与参数管理](./ch05_2_registration_params.md) | `ex2_registration.py` | 打印 `ex2 OK`；能口头解释「为什么 list 装的层优化器看不见」 | 45 min |
| 3 | [5.3 Linear](./ch05_3_Linear.md) + [5.4 shape 推导](./ch05_4_shape_drills.md) | `ex3_shapes.py` | 打印 `ex3 OK`；**所有预测先笔算、再运行**，预测对 ≥ 4/5 | 1 h |
| 4 | [5.5 train/eval 与损失](./ch05_5_train_eval_loss.md) §1–3 | `ex4_train_eval.py` | 打印 `ex4 OK`；能说出哪两类层受 `eval()` 影响 | 45 min |
| 5 | [5.5](./ch05_5_train_eval_loss.md) §4 | `ex5_train_mlp.py` | 打印 `ex5 OK`；有 ReLU > 0.9，无 ReLU ≈ 0.5；回答文件里的思考题 | 1 h |
| 6 | 手册 [ch05_nn_module.md](./ch05_nn_module.md) 里的「5.7 小 CNN 模板」 | `ex6_small_cnn.py` | 打印 `ex6 OK`；**手算**参数量与实际一致 | 1 h |

总计约 5–6 小时，可分 2–3 天。**每天最多做 2 个任务**，做完当天回头改一遍错题。

### 怎么运行

```bash
source ~/venvs/dl/bin/activate
cd learn_pytorch_and_numpy/ex05
python ex1_module_basics.py
```

没填完时报错是正常的；第一个报错的位置就是你下一步要补的地方。

### 卡住了怎么办

1. 先看报错最后一行，再看第几行。
2. 在报错前一行加 `print(x.shape)`、`print(model)`。
3. 卡 15 分钟再看 `ex05/solutions/` 的参考解；**看完关掉，自己重写一遍**。

---

## 通关标准（第 5 章结束的自测）

不看任何资料，在白纸 / 空文件里完成：

1. 写一个 `Net`：`[B,1,28,28] → Conv → ReLU → MaxPool → Flatten → Linear → logits`，并说出每步 shape。
2. 写 20 行训练循环，包含 `train()/eval()`、`zero_grad`、`backward`、`step`。
3. 回答：loss 不下降，你先检查哪三件事？
   - 参考：①`len(list(model.parameters()))` 是否为 0（是否注册）；②是否漏了 `zero_grad/backward/step`；③标签类型与损失是否匹配 / 有没有多余的 softmax。

三题都能做，进入第 6 章（优化器）。

---

## 之后要做什么

- 速查手册里的「了解」级内容（`Embedding`、`LSTM`）**留到真需要时**再学，不占用本章时间。
- 第 6、7、8 章是优化器、数据加载、完整训练流程；到时 ex5 的训练循环会被拆成 `DataLoader` + 验证集版本。
