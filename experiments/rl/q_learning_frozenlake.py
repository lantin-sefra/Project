"""表格型 Q-learning @ FrozenLake（自写核心，gymnasium 只提供环境）。

用法（CPU 即可，全部跑完约 2 分钟）：
    /home/rwr/venvs/dl/bin/python q_learning_frozenlake.py            # 跑全部
    /home/rwr/venvs/dl/bin/python q_learning_frozenlake.py det        # 实验 1：确定性
    /home/rwr/venvs/dl/bin/python q_learning_frozenlake.py slip       # 实验 2：滑冰 + 3 seed
    /home/rwr/venvs/dl/bin/python q_learning_frozenlake.py vi         # 实验 3：值迭代对照
    /home/rwr/venvs/dl/bin/python q_learning_frozenlake.py table      # 实验 4：Q 表为什么不够用

对应笔记：notes/RL_note/ch01_q_learning_frozenlake.md
"""
import sys

import gymnasium as gym
import numpy as np

ARROWS = "←↓→↑"  # FrozenLake 动作编码：0 左 1 下 2 右 3 上


# ---------------------------------------------------------------- 核心算法
def epsilon_greedy(Q, s, eps, rng):
    """以 eps 概率随机探索，否则取 argmax；平局随机打破（全 0 初始化时很重要）。"""
    n_actions = Q.shape[1]
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


def td_update(Q, s, a, r, s_next, terminated, alpha, gamma):
    """Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]。

    关键：终止态没有未来，bootstrap 项必须乘 (1 - terminated)；
    truncated（时间到）不是真终止，仍应 bootstrap，所以这里只看 terminated。
    """
    target = r + gamma * (0.0 if terminated else Q[s_next].max())
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return td_error


def train_q_learning(env, episodes, alpha=0.1, gamma=0.99,
                     eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5, seed=0):
    """eps 线性衰减：前 eps_decay_frac * episodes 个回合从 eps_start 降到 eps_end。"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(episodes)
    decay_eps = max(1, int(episodes * eps_decay_frac))
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        s, _ = env.reset(seed=seed * 100003 + ep if ep == 0 else None)
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            td_update(Q, s, a, r, s_next, terminated, alpha, gamma)
            s = s_next
            done = terminated or truncated
        returns[ep] = r  # FrozenLake 只有到终点才有 r=1，最后一步 r 即回合回报
    return Q, returns


def evaluate_greedy(env, Q, episodes=2000, seed=12345):
    """纯贪心（eps=0）评估成功率。"""
    rng = np.random.default_rng(seed)
    wins = 0
    for i in range(episodes):
        s, _ = env.reset(seed=seed if i == 0 else None)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(epsilon_greedy(Q, s, 0.0, rng))
            done = terminated or truncated
        wins += r
    return wins / episodes


def value_iteration(env, gamma=0.99, tol=1e-10):
    """用环境的真实转移 P 做动态规划，得到 Q*，作为 Q-learning 的「标准答案」。
    只用于对照验收；Q-learning 本身不依赖 P（model-free）。"""
    P = env.unwrapped.P
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    while True:
        V = Q.max(axis=1)
        Q_new = np.zeros_like(Q)
        for s in range(nS):
            for a in range(nA):
                for p, s2, r, term in P[s][a]:
                    Q_new[s, a] += p * (r + gamma * (0.0 if term else V[s2]))
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new


def show_policy(Q, desc, nrow, ncol):
    for i in range(nrow):
        row = ""
        for j in range(ncol):
            ch = desc[i][j]
            row += ch if ch in "HG" else ARROWS[int(np.argmax(Q[i * ncol + j]))]
        print("   ", row)


def make_env(slippery, map_name="4x4"):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)


# ---------------------------------------------------------------- 实验
def exp_det():
    print("== 实验 1：确定性 FrozenLake（is_slippery=False） ==")
    env = make_env(False)
    Q, returns = train_q_learning(env, episodes=2000, alpha=0.5, gamma=0.99, seed=0)
    print("训练最后 200 回合平均回报:", returns[-200:].mean())
    print("greedy 成功率:", evaluate_greedy(env, Q, 200))
    print("策略（箭头=argmax Q；H 洞 G 终点）:")
    show_policy(Q, env.unwrapped.desc.astype(str), 4, 4)
    print("起点 Q[0]:", np.round(Q[0], 4), " 期望最优 ≈ gamma^5 =", round(0.99 ** 5, 4))


def exp_slip():
    print("\n== 实验 2：滑冰 FrozenLake（is_slippery=True），3 个 seed ==")
    env = make_env(True)
    Q_star = value_iteration(env)
    opt_succ = evaluate_greedy(env, Q_star, 5000)
    print(f"值迭代最优策略成功率（天花板）: {opt_succ:.3f}  （理论约 0.74）")
    for seed in range(3):
        Q, returns = train_q_learning(env, episodes=30000, alpha=0.1, gamma=0.99,
                                      eps_end=0.05, seed=seed)
        win = evaluate_greedy(env, Q, 5000)
        curve = [returns[i:i + 5000].mean() for i in range(0, 30000, 5000)]
        print(f"seed={seed}  greedy 成功率={win:.3f}  训练曲线(每5000回合均值)=",
              np.round(curve, 3))
    print("seed=2 的策略 vs 最优策略:")
    print("  Q-learning:")
    show_policy(Q, env.unwrapped.desc.astype(str), 4, 4)
    print("  值迭代 Q*:")
    show_policy(Q_star, env.unwrapped.desc.astype(str), 4, 4)


def exp_vi():
    print("\n== 实验 3：Q-learning 收敛到 Q* 吗？（滑冰，更多回合 + 较小 alpha） ==")
    env = make_env(True)
    Q_star = value_iteration(env)
    for episodes in (5000, 30000, 100000):
        Q, _ = train_q_learning(env, episodes=episodes, alpha=0.05, gamma=0.99, seed=0)
        err = np.abs(Q - Q_star)[:-1].max()  # 终点那行恒为 0，无所谓
        print(f"episodes={episodes:>6}  max|Q-Q*|={err:.3f}  "
              f"V(start): Q-learning={Q[0].max():.3f}  Q*={Q_star[0].max():.3f}")


def exp_table():
    print("\n== 实验 4：为什么表格不够用（只点到为止） ==")
    # 4a：状态数随地图增长
    for name in ("4x4", "8x8"):
        env = make_env(True, name)
        print(f"FrozenLake {name}: |S|={env.observation_space.n}, "
              f"|A|={env.action_space.n}, Q 表 {env.observation_space.n * 4} 个数")
    # 4b：CartPole 状态是 4 维连续量，表格只能先离散化
    env = gym.make("CartPole-v1")
    low = np.array([-2.4, -3.0, -0.21, -3.5])
    high = np.array([2.4, 3.0, 0.21, 3.5])
    rng = np.random.default_rng(0)
    for bins in (6, 10, 20):
        edges = [np.linspace(low[i], high[i], bins + 1)[1:-1] for i in range(4)]
        visited = set()
        for ep in range(300):
            s, _ = env.reset(seed=ep)
            done = False
            while not done:
                visited.add(tuple(int(np.digitize(s[i], edges[i])) for i in range(4)))
                s, _, term, trunc, _ = env.step(int(rng.integers(2)))
                done = term or trunc
        total = bins ** 4
        print(f"CartPole 每维 {bins:>2} 格: 表格状态数 {total:>6}, "
              f"随机策略 300 回合只访问到 {len(visited):>5} ({len(visited) / total:.1%})")
    print("→ 维度一高：表爆炸 + 大多数格子从没被更新过 + 相邻格子互不共享信息。")
    print("  解决思路：用带参数 θ 的函数 Q_θ(s,a) 做泛化（下一站 DQN）。")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    table = {"det": exp_det, "slip": exp_slip, "vi": exp_vi, "table": exp_table}
    if which == "all":
        for f in table.values():
            f()
    else:
        table[which]()
