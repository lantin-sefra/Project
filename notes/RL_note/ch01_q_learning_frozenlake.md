# RL 第 1 章 · 表格型 Q-learning（FrozenLake）

> **设备提示**：本章全部 NumPy + gymnasium，**CPU 即可**，不需要 GPU。解释器用 `/home/rwr/venvs/dl/bin/python`（gymnasium 1.3.0）。  
> 从下一章（DQN）起才会用到 PyTorch；若届时要用 GPU，再设 `export HSA_OVERRIDE_GFX_VERSION=11.0.0`，小网络依然优先 CPU。

对应「七阶段」第 2 步 / 14 天冲刺 Day 1–2。本章目标：**自己写出一个能收敛的 Q-learning**，并且能解释清楚每一行为什么这样写。  
下一章 DQN 的全部零件（ε-greedy、TD 目标、终止态处理）都在这里先练熟，只是把 `Q[s, a]` 这张表换成神经网络。

📎 理论（先扫一遍再写代码，卡住再回头精读）：

- **《动手学深度学习》**（D2L）强化学习章：
  - [17.1 马尔可夫决策过程](https://zh.d2l.ai/chapter_reinforcement-learning/mdp.html)（MDP、回报、折扣、价值函数）
  - [17.2 值迭代](https://zh.d2l.ai/chapter_reinforcement-learning/value-iter.html)（Bellman 最优方程；本章实验 3 的「标准答案」就是它）
  - [17.3 Q-learning](https://zh.d2l.ai/chapter_reinforcement-learning/qlearning.html)（本章主算法，D2L 里也是 FrozenLake）
  - 若中文站缺页，英文原版同路径：[d2l.ai/chapter_reinforcement-learning](https://d2l.ai/chapter_reinforcement-learning/qlearning.html)
- **《蘑菇书》EasyRL**（[在线版](https://datawhalechina.github.io/easy-rl/)）：
  - 第 1 章 绪论（智能体/环境/回报/探索与利用）
  - 第 2 章 马尔可夫决策过程（MP → MRP → MDP；Bellman 方程）
  - **第 3 章 表格型方法**（3.1 MDP / 3.2 Q 表格 / 3.3 免模型预测 / 3.4 Sarsa / 3.5 Q-learning ← 本章主读；3.6 on-policy vs off-policy）
  - 第 6 章 深度 Q 网络（下一章预习，今天只读开头「为什么需要函数逼近」一节）
- 对照阅读（可选）：
  - Sutton & Barto《Reinforcement Learning》第 3 章（MDP）、**6.5 Q-learning: Off-policy TD Control**
  - [Spinning Up · Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)（符号/术语，后面 SAC 也用这套）
  - [CleanRL · DQN](https://docs.cleanrl.dev/rl-algorithms/dqn/)（下一章对照对象，今天不用读代码）
  - [Gymnasium · FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)（环境的动作/观测/奖励定义）

🧪 代码：[`experiments/rl/q_learning_frozenlake.py`](../../experiments/rl/q_learning_frozenlake.py)

---

## 今日目标与验收清单

**目标**：不看框架，写出表格型 Q-learning，在 FrozenLake 上学到接近最优的策略，并说清为什么需要 DQN。

**验收（全部打勾才算今天过关）**：

- [ ] **复述**（不看笔记，能口头/手写）：MDP 五元组 \((S,A,P,R,\gamma)\)；\(Q^\pi\) 与 \(Q^*\) 的定义；Q-learning 更新式；ε-greedy；为什么 Q-learning 是 off-policy。
- [ ] **先预测再跑**：跑实验前，在纸上写下「确定性 4×4 的起点 \(Q^*(s_0, \text{最优动作})\) 是多少」（提示：最短路径 6 步还是 5 步？到终点才有 +1），再对照实验 1 输出。
- [ ] **自写核心**：`epsilon_greedy` / `td_update` / `train_q_learning` 三个函数你能独立默写出骨架；说得出 `terminated` 与 `truncated` 在 TD 目标里为什么区别对待。
- [ ] **实验 1**：确定性 FrozenLake，greedy 成功率 = 100%，画出的箭头策略能从 S 走到 G。
- [ ] **实验 2**：滑冰 FrozenLake，3 个 seed 的 greedy 成功率都 ≥ 0.70（最优天花板 ≈ 0.74）。
- [ ] **实验 3**：能解释「训练曲线只有 ≈0.53，greedy 评估却有 ≈0.74」。
- [ ] **衔接**：能用一句话回答「Q 表在 CartPole 上为什么不行」，并写出 DQN 相对本章的两处改动（神经网络 + 经验回放/目标网络，见 1.8）。

预计用时：理论 1.5–2 h，写/跑代码 1.5–2 h。

---

## 1.1 MDP 与回报：强化学习的「题目格式」

**是什么**：智能体在每个时刻 \(t\) 观察状态 \(s_t\)，选动作 \(a_t\)，环境返回奖励 \(r_{t+1}\) 和下一状态 \(s_{t+1}\)。MDP 把这件事形式化为 \((S, A, P, R, \gamma)\)：

- \(P(s'|s,a)\)：转移概率（FrozenLake 滑冰版里**不是确定的**）
- \(R\)：奖励
- \(\gamma\in[0,1)\)：折扣因子
- **马尔可夫性**：下一状态只依赖当前 \((s,a)\)，与更早历史无关

目标：最大化**折扣回报** \(G_t=\sum_{k\ge 0}\gamma^k r_{t+k+1}\) 的期望。

**FrozenLake 4×4 的具体设定**（gymnasium `FrozenLake-v1`）：

```text
S F F F        S 起点  F 冰面（安全）
F H F H        H 洞（掉进去回合结束，奖励 0）
F F F H        G 终点（到了回合结束，奖励 +1）
H F F G        其余一切奖励都是 0  ← 稀疏奖励
```

| 项 | 取值 |
|----|------|
| 状态 | 0–15，按行主序编号 `s = 行*4 + 列` |
| 动作 | 0 左 · 1 下 · 2 右 · 3 上 |
| 奖励 | 只有到 G 时 +1 |
| `is_slippery=True` | 你选的方向只有 1/3 概率生效，另外 2/3 概率滑向两侧垂直方向（所以 P 是随机的） |
| 回合终止 | `terminated`：掉洞或到 G；`truncated`：超过 100 步（`TimeLimit`） |

**怎么用（看环境）**：

```python
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
s, info = env.reset(seed=0)
s2, r, terminated, truncated, info = env.step(2)   # 向右
print(s, s2, r, terminated, truncated)
# 0 1 0.0 False False
```

**易错点**：

1. gymnasium 的 `step` 返回 **5 个值**（旧 gym 是 4 个，`done` 被拆成 `terminated` 和 `truncated`）。网上旧教程直接抄会报错。
2. `terminated` 与 `truncated` **含义不同**，TD 目标里处理方式也不同（见 1.4）。
3. 稀疏奖励：前期几乎全是 0，探索不够 Q 表永远学不到东西。

---

## 1.2 价值函数与 Bellman 方程

**是什么**：

- 动作价值 \(Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]\)：在 \(s\) 先做 \(a\)，之后按 \(\pi\) 行动的期望回报。
- 最优动作价值 \(Q^*(s,a)=\max_\pi Q^\pi(s,a)\)，满足 **Bellman 最优方程**：

\[
Q^*(s,a)=\sum_{s'}P(s'|s,a)\Big[r+\gamma\max_{a'}Q^*(s',a')\Big]
\]

知道 \(Q^*\) 就知道最优策略：\(\pi^*(s)=\arg\max_a Q^*(s,a)\)。

**两条路求 \(Q^*\)**：

| | 值迭代（DP） | Q-learning（TD） |
|---|---|---|
| 需要 \(P\) 吗 | **需要**（model-based） | **不需要**（model-free，只靠采样） |
| 做法 | 把 Bellman 方程当赋值反复迭代 | 用一条条真实 \((s,a,r,s')\) 样本做近似迭代 |
| 本章用途 | 实验 3，当「标准答案」 | 主角 |

代码里值迭代只有十行：

```python
Q_new[s, a] += p * (r + gamma * (0.0 if term else V[s2]))   # 对 P[s][a] 里每个 (p, s2, r, term) 累加
```

这里你能直接看到 Q-learning 更新式其实就是「把期望 \(\sum_{s'}P(\cdot)\) 换成一次采样」。

**易错点**：把 \(V\) 与 \(Q\) 搞混。\(V(s)=\max_a Q(s,a)\)（最优情形）；表格存 \(Q\) 而不是 \(V\) 是因为做决策时不需要知道 \(P\)。

---

## 1.3 ε-greedy：探索与利用

**是什么**：以概率 \(\varepsilon\) 随机选动作（探索），否则选 \(\arg\max_a Q(s,a)\)（利用）。

```python
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    best = np.flatnonzero(q == q.max())     # 平局随机打破
    return int(rng.choice(best))
```

**ε 调度**：本实验从 1.0 线性降到 0.05（前一半回合内降完）。DQN 里同一套。

**易错点**：

1. **`np.argmax` 平局永远返回第一个**。Q 表全 0 初始化时，所有动作并列，argmax 恒选「左」，智能体会反复撞墙、几乎学不到东西（可以自己改成 `argmax` 对比实验）。所以要随机打破平局。
2. ε 衰减太快 → 还没找到过一次终点就几乎不探索了；永远不衰减 → 评估时被噪声拖累。
3. 评估（greedy）要把 ε 设成 0，训练曲线与评估成绩是两回事（见实验 3 的解读）。

---

## 1.4 Q-learning 的 TD 更新（本章核心）

**是什么**：看到一条经验 \((s,a,r,s')\) 就更新一次：

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\underbrace{r+\gamma\max_{a'}Q(s',a')}_{\text{TD 目标}}-Q(s,a)\Big]
\]

方括号整体叫 **TD 误差** \(\delta\)。直觉：用「走一步之后更准的估计」去修正「当前的估计」，修正幅度由学习率 \(\alpha\) 控制。（这就是**自举 bootstrapping**。）

**怎么用**：

```python
def td_update(Q, s, a, r, s_next, terminated, alpha, gamma):
    target = r + gamma * (0.0 if terminated else Q[s_next].max())
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return td_error
```

**为什么是 off-policy**：行为策略是 ε-greedy（会乱走），但目标里用的是 \(\max_{a'}\)，即**贪心策略**的价值。「采样用的策略」与「被评估/改进的策略」不是同一个。对比 Sarsa 的目标 \(r+\gamma Q(s',a')\)（\(a'\) 是实际选的动作，on-policy）——EasyRL 3.4–3.6 有悬崖行走的对比例子。off-policy 是后面**经验回放**能成立的前提。

**易错点**：

1. **终止态必须不 bootstrap**：到 G / 掉进 H 后，未来回报就是 0，目标是 \(r\) 而不是 \(r+\gamma\max Q(s')\)。漏掉的后果：终止态的 Q 值乱长，Q 表整体偏高。DQN 里对应 `(1 - done)` 这一项，同样是头号 bug。
2. **`truncated`（超时）≠ 终止**：超时只是被强行截断，世界并没有结束，仍该 bootstrap。所以代码里只用 `terminated` 判断；DQN 同理。
3. \(\alpha\) 太大 → 随机环境里 Q 值抖动不收敛；太小 → 学得慢。滑冰版用 0.05–0.1，确定性版可以到 0.5。
4. \(\gamma\) 决定「走得快」的偏好：\(\gamma<1\) 时到终点的步数越少回报越高，所以最优起点 Q 值 = \(\gamma^{\text{步数}-1}\)（实验 1 用它验证）。

---

## 1.5 训练循环

```python
for ep in range(episodes):
    eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
    s, _ = env.reset()
    done = False
    while not done:
        a = epsilon_greedy(Q, s, eps, rng)
        s_next, r, terminated, truncated, _ = env.step(a)
        td_update(Q, s, a, r, s_next, terminated, alpha, gamma)
        s = s_next
        done = terminated or truncated
```

和第 6 章（`optim` 训练主链）对照着看：`td_update` ≈ 「前向算 loss → 反传 → step」被手写成一行；DQN 里这一行会变成真正的 `loss.backward(); optimizer.step()`。

---

## 1.6 逐步实验（每步：先预测，再跑，再对照）

```bash
cd /home/rwr/Project/experiments/rl
/home/rwr/venvs/dl/bin/python q_learning_frozenlake.py det     # 实验 1，约 1 s
/home/rwr/venvs/dl/bin/python q_learning_frozenlake.py slip    # 实验 2，约 40 s
/home/rwr/venvs/dl/bin/python q_learning_frozenlake.py vi      # 实验 3，约 70 s
/home/rwr/venvs/dl/bin/python q_learning_frozenlake.py table   # 实验 4，约 5 s
# 不带参数 = 全部（约 2 分钟）
```

下面「实测输出」是在本机实际跑出来的（seed 固定，你重跑应基本一致）。

### 实验 1 · 确定性 FrozenLake（`is_slippery=False`）

**预测**：S 到 G 最短路径 6 步（右右下下下右），最后一步才 +1，起点最优动作的 \(Q^*=\gamma^{5}\)（前 5 步 0 奖励，第 6 步 1，折扣 \(0.99^5\)）。

**实测**：

```text
训练最后 200 回合平均回报: 0.94
greedy 成功率: 1.0
策略:
    ↓→↓←
    ↓H↓H
    →↓↓H
    H→→G
起点 Q[0]: [0.9415 0.951  0.951  0.9415]  期望最优 ≈ gamma^5 = 0.951
```

**怎么读**：

- 起点的「下」「右」两个动作并列最优 0.951（两条最短路径），与 \(\gamma^5\) 吻合 → TD 更新与终止态处理写对了。
- 训练平均回报 0.94 < 1：ε=0.05 的随机动作偶尔把它推进洞里，greedy 评估才是 1.0。
- 箭头策略里某些格子（如第一行第 4 个 `←`）是「没走到过、Q 全 0」的格子，随便取一个动作，不影响最优路径。

### 实验 2 · 滑冰 FrozenLake（`is_slippery=True`），3 个 seed

**预测**：随机转移下，即使最优策略也不可能 100% 成功。你猜最优成功率是多少？（先写下，再往下看。）

**实测**：

```text
值迭代最优策略成功率（天花板）: 0.743
seed=0  greedy 成功率=0.742  训练曲线(每5000回合均值)= [0.024 0.062 0.249 0.532 0.518 0.524]
seed=1  greedy 成功率=0.740  训练曲线(每5000回合均值)= [0.024 0.069 0.249 0.524 0.524 0.536]
seed=2  greedy 成功率=0.742  训练曲线(每5000回合均值)= [0.023 0.067 0.263 0.529 0.52  0.538]
```

策略对比（seed=2）——几乎一模一样，只有 (1,2) 一格不同，且该格不在最优策略的主要访问路径上，对成功率影响可忽略：

```text
Q-learning:      值迭代 Q*:
    ←↑↑↑            ←↑↑↑
    ←H→H            ←H←H
    ↑↓←H            ↑↓←H
    H→↓G            H→↓G
```

**怎么读**：

- 最优策略的箭头看起来很「反直觉」：起点 `←` 靠墙、很多格子朝着背离终点的方向走。原因：滑冰时选 `←` 实际会在「上/下/左」之间随机，**撞墙留在原地**是在避免滑进洞里。表格学出来了这种「用墙当护栏」的策略，这是一个很好的 Q-learning 直觉练习。
- 3 个 seed 都 ≈ 0.74 = 天花板，验收达标（≥ 0.70）。
- 训练曲线前期 0.02 → 后期 0.52：稀疏奖励下前 5000 回合基本是在碰运气找终点；一旦找到，价值沿着 Bellman 链**逐格反向传播**回起点，曲线才抬升。

### 实验 3 · 与标准答案 \(Q^*\) 对比 & 训练/评估为什么不一样

**预测**：训练平均回报只有 ≈0.53，而 greedy 评估有 ≈0.74——你觉得差的 0.2 去哪了？

**实测**（\(\alpha=0.05\)，seed=0）：

```text
episodes=  5000  max|Q-Q*|=0.207  V(start): Q-learning=0.574  Q*=0.542
episodes= 30000  max|Q-Q*|=0.126  V(start): Q-learning=0.556  Q*=0.542
episodes=100000  max|Q-Q*|=0.106  V(start): Q-learning=0.516  Q*=0.542
```

**怎么读**：

- **训练 0.53 vs 评估 0.74**：训练时每步还有 \(\varepsilon=0.05\) 的随机动作，而在「靠墙护栏」这类策略里一步随机动作就可能掉洞，所以训练回报系统性偏低；评估把 ε 关掉。（实测验证：同一张训练好的 Q 表，greedy 成功率 0.749，若评估时仍保留 ε=0.05 则降到 0.548，与训练曲线的 ≈0.53 吻合；平均每回合约 44 步，所以 0.05 的噪声累计起来影响很大。）这是 **行为策略 ≠ 目标策略**（off-policy）最直观的数字体现。
- `V(start)` 在 0.52–0.57 间摆动、`max|Q-Q*|` 没有降到 0：固定学习率 \(\alpha\) 下，Q 值会在真值附近**持续抖动**（随机环境里每个样本都有噪声，α 不衰减就不会静止）。想要严格收敛需要 \(\alpha_t\) 衰减（Robbins–Monro 条件，EasyRL/Sutton 6.5 有讲），工程上我们只关心**策略**已经最优。
- `max|Q-Q*|` 里最大误差通常出现在**很少访问**的 (s,a) 上——这正是表格方法的弱点：没访问到的格子就是没学到。

### 实验 4 · 为什么表格不够（点到为止，今天不写 DQN）

**实测**：

```text
FrozenLake 4x4: |S|=16, |A|=4, Q 表 64 个数
FrozenLake 8x8: |S|=64, |A|=4, Q 表 256 个数
CartPole 每维  6 格: 表格状态数   1296, 随机策略 300 回合只访问到    81 (6.2%)
CartPole 每维 10 格: 表格状态数  10000, 随机策略 300 回合只访问到   219 (2.2%)
CartPole 每维 20 格: 表格状态数 160000, 随机策略 300 回合只访问到   808 (0.5%)
```

**怎么读**：

1. **维度灾难**：CartPole 的状态是 4 个连续数（位置、速度、角度、角速度），表格只能先切格子；每维 \(k\) 格就是 \(k^4\) 个状态，随维度指数增长。边缘调度的状态（队列长度、各节点负载、带宽……）往往几十维，表格根本无从谈起。
2. **大多数格子从未被访问**：覆盖率随格数急剧下降，没访问到的格子 Q 没被更新过。
3. **不泛化**：相邻两格在物理上几乎一样，但表格里互不相干，访问过一格学到的东西帮不了邻居。

→ 解决思路：用带参数的 \(Q_\theta(s,a)\)（神经网络）代替表格，**相似状态共享参数、一次更新惠及一片状态**。这就是 DQN 要做的事（下一章）。

---

## 1.7 常见坑清单

| 现象 | 多半是 | 怎么查 |
|------|--------|--------|
| 学不到任何东西、greedy 成功率 0 | argmax 平局总选第一个；ε 没衰减/太早衰减；回合数太少 | 打印训练回报曲线；换成随机打破平局 |
| 起点 Q 值 > 1 或乱涨 | 终止态仍 bootstrap | 检查 `(1 - terminated)` |
| 报错 `too many values to unpack` | 用了旧 gym 的 4 返回值写法 | gymnasium 是 5 个返回值 |
| 确定性版收敛但滑冰版不收敛 | α 太大 / 回合数不足 | 减小 α（0.05–0.1）、拉长训练 |
| 评估成绩远高于训练曲线 | 正常：训练含探索噪声 | 用 ε=0 单独评估 |
| 结果跑一次一个样 | 随机种子 | 至少 3 个 seed 报均值（DQN/SAC 同样如此） |
| 把 `truncated` 当终止 | TimeLimit 截断被当成「真结束」 | 只用 `terminated` 判断是否 bootstrap |

---

## 1.8 与下一站 DQN 的衔接（只看，不写）

| | 本章（表格 Q-learning） | 下一章（DQN） |
|---|---|---|
| Q 的载体 | `Q[s, a]` 表 | 神经网络 \(Q_\theta(s)\to\mathbb R^{|A|}\) |
| 更新 | `Q[s,a] += α·δ` | 最小化 \((y-Q_\theta(s,a))^2\)，`loss.backward(); optimizer.step()` |
| 目标 \(y\) | \(r+\gamma\max_{a'}Q(s',a')\) | 同样的 \(y\)，但用**目标网络** \(Q_{\theta^-}\) 计算，且 `.detach()`（回看第 4 章 4.5 节） |
| 数据 | 一条经验用一次、立即丢弃 | **经验回放**：存起来随机采样，打破样本间相关性（off-policy 才允许这么做） |
| 终止处理 | `(1 - terminated)` | 完全一样 |
| ε-greedy | 一样 | 一样 |

为什么 DQN 需要「经验回放 + 目标网络」：函数逼近 + 自举 + off-policy 三者同时出现（「致命三角」）时训练容易发散；这两个技巧就是为它兜底。详见 EasyRL 第 6 章、[DQN 论文](https://www.nature.com/articles/nature14236)。**今天只需要知道动机，不要开始写。**

**DQN 验收预告（下一单元用）**：自写 DQN @ CartPole-v1，3 个 seed，评估回报接近 500（上限）。

---

## 本章小结

| 主题 | 记住 |
|------|------|
| MDP | \((S,A,P,R,\gamma)\)；目标是最大化折扣回报的期望 |
| Q\* 与策略 | 取 \(\arg\max_a Q^*(s,a)\) 即最优策略；Bellman 最优方程是核心 |
| ε-greedy | 训练时探索，评估时 ε=0；平局随机打破 |
| TD 更新 | \(Q\leftarrow Q+\alpha[r+\gamma\max Q(s')-Q]\)；终止态不 bootstrap；`truncated` 仍 bootstrap |
| off-policy | 采样用 ε-greedy，目标用 max；为经验回放铺路 |
| 表格局限 | 维度灾难 + 访问稀疏 + 不泛化 → 函数逼近（DQN） |

## 练习

1. 把 `epsilon_greedy` 里的随机打破平局换成 `np.argmax`，重跑实验 1，观察现象并解释。
2. 去掉 `td_update` 里的 `terminated` 判断（始终 bootstrap），看实验 1 的起点 \(Q\) 值变成多少，为什么？
3. 滑冰版把 \(\alpha\) 改成 0.5，观察 greedy 成功率与 `max|Q-Q*|` 的变化。
4. 把 \(\gamma\) 改成 0.9 / 1.0，策略与起点 Q 值如何变？（提示：\(\gamma=1\) 时「走得快」不再有额外好处。）
5. （选做）在 `8x8` 滑冰地图上跑同样的算法，需要多少回合才能出现非零的 greedy 成功率？这对稀疏奖励意味着什么？
6. （选做，写 DQN 前的热身）实现 Sarsa（把 `max Q[s']` 换成实际选的 `Q[s', a']`），在 `slip` 实验上与 Q-learning 对比。
