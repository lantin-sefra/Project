"""两层网络的手写反向传播 + 方向导数梯度检验。

约定:每列一个样本。
    Z1 = W1 X + b1 1^T,  H = tanh(Z1),  Z2 = W2 H + b2 1^T
    L  = 1/2 ||Z2 - Y||_F^2
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- 数据与参数 ----------
n_in, n_hid, n_out, p = 4, 5, 3, 8          # 输入维、隐层维、输出维、样本数
X = rng.standard_normal((n_in, p))
Y = rng.standard_normal((n_out, p))

params = {
    "W1": rng.standard_normal((n_hid, n_in)) * 0.5,
    "b1": rng.standard_normal(n_hid) * 0.1,
    "W2": rng.standard_normal((n_out, n_hid)) * 0.5,
    "b2": rng.standard_normal(n_out) * 0.1,
}


# ---------- 前向 ----------
def forward(P):
    Z1 = P["W1"] @ X + P["b1"][:, None]      # 广播 = b1 1^T
    H = np.tanh(Z1)
    Z2 = P["W2"] @ H + P["b2"][:, None]
    L = 0.5 * np.sum((Z2 - Y) ** 2)          # 0.5 * ||.||_F^2
    return L, (Z1, H, Z2)


# ---------- 反向 ----------
def backward(P, cache):
    Z1, H, Z2 = cache
    G2 = Z2 - Y                               # dL/dZ2
    grads = {}
    grads["W2"] = G2 @ H.T                    # G X^T 规则
    grads["b2"] = G2.sum(axis=1)              # G 1:广播对应求和
    GH = P["W2"].T @ G2                       # A^T G 规则
    G1 = GH * (1 - H ** 2)                    # Hadamard 积 * tanh'
    grads["W1"] = G1 @ X.T
    grads["b1"] = G1.sum(axis=1)
    return grads


# ---------- 梯度检验 ----------
def directional_check(P, grads, eps=1e-6, trials=3):
    """随机方向 D:验证 (L(P+eps D) - L(P-eps D)) / 2eps ≈ <grad, D>_F。"""
    for t in range(trials):
        D = {k: rng.standard_normal(v.shape) for k, v in P.items()}
        Pp = {k: P[k] + eps * D[k] for k in P}
        Pm = {k: P[k] - eps * D[k] for k in P}
        num = (forward(Pp)[0] - forward(Pm)[0]) / (2 * eps)
        ana = sum(np.sum(grads[k] * D[k]) for k in P)   # Frobenius 内积之和
        rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
        print(f"方向检验 {t + 1}: 数值={num:+.8f}  解析={ana:+.8f}  相对误差={rel:.2e}")


def elementwise_check(P, grads, eps=1e-6):
    """逐元素中心差分,报告每个参数的最大绝对误差。"""
    for k in P:
        num = np.zeros_like(P[k])
        it = np.nditer(P[k], flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = P[k][idx]
            P[k][idx] = old + eps
            Lp = forward(P)[0]
            P[k][idx] = old - eps
            Lm = forward(P)[0]
            P[k][idx] = old
            num[idx] = (Lp - Lm) / (2 * eps)
        print(f"逐元素检验 {k}: 最大绝对误差 = {np.max(np.abs(num - grads[k])):.2e}")


if __name__ == "__main__":
    L, cache = forward(params)
    grads = backward(params, cache)
    print(f"L = {L:.6f}\n")
    directional_check(params, grads)
    print()
    elementwise_check(params, grads)

    # 简单训练几步,确认损失下降
    print("\n梯度下降:")
    lr = 0.01
    for step in range(201):
        L, cache = forward(params)
        grads = backward(params, cache)
        for k in params:
            params[k] -= lr * grads[k]
        if step % 50 == 0:
            print(f"step {step:3d}  L = {L:.6f}")
