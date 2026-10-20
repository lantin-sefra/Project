**Shape 和 Attributes 讲解**
这两段是在说输入输出长什么样，以及 `weight`、`bias` 一开始怎么填随机数。符号拆开就是下面这样。

#### Shape：只改最后一维，前面的维度原样保留

\[
(*, H_{in}) \xrightarrow{LIN} (*, H_{out})
\]

- \(H_{in}\) 就是构造时的 `in_features`。
- \(H_{out}\) 就是 `out_features`。
- \(*\) 表示最后一维前面可以有任意多个维度，也可以一个都没有。这些维度原样保留。

`nn.Linear(20, 30)` 时，\(H_{in}=20\)，\(H_{out}=30\)：

| 输入 | 输出 |
|---|---|
| `[20]` | `[30]` |
| `[128, 20]` | `[128, 30]` |
| `[8, 16, 20]` | `[8, 16, 30]` |

```python
LIN = nn.Linear(20, 30)
LIN(torch.randn(128, 20)).shape      # [128, 30]
LIN(torch.randn(8, 16, 20)).shape    # [8, 16, 30]  序列也能接
```

最后一维必须等于 20，否则对不上 `weight` 的列数。

#### Attributes：两个可学习张量

\[
\text{weight}:\ (\text{out\_features},\ \text{in\_features})
\qquad
\text{bias}:\ (\text{out\_features})
\]

`Linear(20, 30)` 就是 `weight` 为 `[30, 20]`，`bias` 为 `[30]`。这就是上一问的 \(y=xA^\top+b\)：\(A\) 按「行 = 输出、列 = 输入」存。

#### 初始化：$\small\mathcal{U}(-\sqrt{k},\sqrt{k})$

\(\mathcal{U}(a,b)\) 是区间 \((a,b)\) 上的均匀分布，每个元素独立抽取。这里

\[
k=\frac{1}{\text{in\_features}},\qquad a=-\sqrt{k},\qquad b=\sqrt{k}
\]

`in_features=20` 时，\(k=1/20=0.05\)，\(\sqrt{k}\approx 0.224\)。所以 `weight` 和 `bias` 的每个数都从大约 \((-0.224,\ 0.224)\) 里均匀抽。

除以输入维度，是为了输入维数变大时，单个输出不至于一开始就特别大。`bias=False` 时没有 `bias`，也就没有这项初始化。





#### 为什么`torch.nn.Linear.weight.shape()` 出来的是[out,in]
Linear 的官方定义就是 \(y = xA^\top + b\)，矩阵 \(A\) 按 `(out_features, in_features)` 存。`Linear(20, 10)` 的 `weight` 因此是 `[10, 20]`。

每一行是一个输出神经元的全部入边权重，而且在内存里是连续的。第 \(i\) 个输出就是这一行和输入的点积：

\[
y_i = A_{i,:}\cdot x + b_i
\]

反传也刚好对齐：\(\partial L/\partial A = (\partial L/\partial y)^\top x\)，形状是 `[10, 32] @ [32, 20] = [10, 20]`，和 `weight` 同形状，可以直接加进 `weight.grad`。
