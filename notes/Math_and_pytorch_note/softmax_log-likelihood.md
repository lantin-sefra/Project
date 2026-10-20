
**链接：**[softmax回归-3.4](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)


### 前提条件（原书写得比较分散）：

上标 \(^{(i)}\) 只表示「第 \(i\) 个样本」，四个向量不是同一个东西：

| 符号 | 含义 | 形状 |
|------|------|------|
| \(\mathbf x^{(i)}\) | 输入特征 | \(d\) |
| \(\mathbf z^{(i)}\) | 线性层输出（logits）。书上批量矩阵写作 \(\mathbf O\)，第 \(i\) 行就是 \(\mathbf z^{(i)}\) | \(q\) |
| \(\hat{\mathbf y}^{(i)}=\mathrm{softmax}(\mathbf z^{(i)})\) | 模型给出的各类**概率** | \(q\) |
| \(\mathbf y^{(i)}\) | **真实标签**（独热：第 \(c\) 类则 \(y_c=1\)，其余为 0） | \(q\) |

单样本：\(\mathbf z=\mathbf W^\top\mathbf x+\mathbf b\)（\(\mathbf W\in\mathbb{R}^{d\times q}\)），再 \(\hat{\mathbf y}=\mathrm{softmax}(\mathbf z)\)。  
softmax 保证 \(\hat y_j>0\)、\(\sum_j\hat y_j=1\)，故 \(\hat y_j=P(Y=j\mid\mathbf x)\)。独热标签下单样本概率就是正确类那一项：\(P(\mathbf y\mid\mathbf x)=\hat y_c\)。

小批量（书 3.4.5）把 \(n\) 个样本按行叠起来：\(\mathbf X\in\mathbb{R}^{n\times d}\)，\(\mathbf W\in\mathbb{R}^{d\times q}\)，\(\mathbf b\in\mathbb{R}^{1\times q}\)（对行广播），

\[
\mathbf Z=\mathbf X\mathbf W+\mathbf b,\qquad
\hat{\mathbf Y}=\mathrm{softmax}(\mathbf Z)\quad\text{（按行）}.
\]

（PyTorch `nn.Linear` 把权重存成 \(W^\top\)，shape `[q, d]`，算出来的 \(\mathbf Z\) 一样。）

样本独立同分布时，数据集的似然是各样本概率的乘积。训练用最大似然：最大化 \(P(\mathbf Y\mid\mathbf X)\)；\(\log\) 严格递增，等价于最小化负对数似然。独热标签下它就是交叉熵。




softmax 给出的是「给定输入，每个类别的条件概率」。把这些概率写成数据集的似然，再取负对数，就得到训练时要最小化的交叉熵。下面按这个顺序证明，并说明它在分类训练里做什么。

## 从 softmax 到单样本概率

设 logits 为 \(z\in\mathbb{R}^q\)，softmax 定义为

\[
\hat y_j=\frac{e^{z_j}}{\sum_{k=1}^{q}e^{z_k}}.
\]

于是 \(\hat y_j>0\) 且 \(\sum_j\hat y_j=1\)，可以把它当成条件分布：

\[
P(Y=j\mid\mathbf x)=\hat y_j.
\]

真实标签若是第 \(c\) 类，独热向量 \(\mathbf y\) 满足 \(y_c=1\)，其余为 0。这个样本的概率就是 \(\hat y_c\)。用乘积写出来是同一件事：

\[
P(\mathbf y\mid\mathbf x)=\prod_{j=1}^{q}\hat y_j^{y_j}=\hat y_c,
\]

因为 \(y_j=0\) 的项都是 \(\hat y_j^0=1\)，只留下 \(\hat y_c^1\)。这就是类别分布（categorical distribution）的概率质量函数。






## 从数据集到负对数似然

\(n\) 个样本若视为独立，联合概率是各样本概率的乘积，即书上的 (3.4.6)：

\[
P(\mathbf Y\mid\mathbf X)=\prod_{i=1}^{n}P(\mathbf y^{(i)}\mid\mathbf x^{(i)})=\prod_{i=1}^{n}\prod_{j=1}^{q}\bigl(\hat y_j^{(i)}\bigr)^{y_j^{(i)}}.
\]

最大似然是让这个数尽量大。对数严格递增，所以最大化 \(P\) 与最大化 \(\log P\) 是同一个优化问题。对数把连乘变成求和：

\[
\log P(\mathbf Y\mid\mathbf X)=\sum_{i=1}^{n}\sum_{j=1}^{q}y_j^{(i)}\log\hat y_j^{(i)}.
\]

优化器习惯做最小化，两边乘 \(-1\) 方向反过来，得到 (3.4.7)：

\[
-\log P(\mathbf Y\mid\mathbf X)=\sum_{i=1}^{n}\Biggl(-\sum_{j=1}^{q}y_j^{(i)}\log\hat y_j^{(i)}\Biggr)=\sum_{i=1}^{n}l(\mathbf y^{(i)},\hat{\mathbf y}^{(i)}).
\]

其中单样本损失就是 (3.4.8)：

\[
l(\mathbf y,\hat{\mathbf y})=-\sum_{j=1}^{q}y_j\log\hat y_j.
\]

独热标签下只剩正确类那一项：

\[
l=-\log\hat y_c=-z_c+\log\sum_{k=1}^{q}e^{z_k}.
\]

\(\hat y_j\in(0,1]\) 时 \(\log\hat y_j\le 0\)，所以 \(l\ge 0\)。\(\hat y_c=1\) 时损失为 0，这是下界。数据有噪声或特征不够时，模型到不了这个下界。

(3.4.8) 也叫交叉熵：分布 \(p,q\) 的交叉熵是 \(-\sum_j p_j\log q_j\)。这里 \(p\) 是独热标签，\(q\) 是 softmax 概率。

## 它用来做什么

分类训练的目标就是最小化这份负对数似然：参数更新的方向，就是让训练集上「真实类别的概率」整体变大。

对 logits 求导很干净。由 \(l=-z_c+\log\sum_k e^{z_k}\) 得

\[
\frac{\partial l}{\partial z_j}=\hat y_j-y_j.
\]

当$j = c$ 时， $y_j = 1$; 否则 $y_j = 0$. 
预测概率高于标签就往下压，低于标签就往上抬。这是 softmax 配交叉熵成为多分类默认损失的原因。

在 PyTorch 里不要先 `softmax` 再手写 \(-\log\)。`nn.CrossEntropyLoss` 直接吃 logits 和类别下标，内部做的就是上面的 \(-\log\hat y_c\)，并用 log-sum-exp 避免 \(e^{z}\) 溢出：

```python
loss_fn = nn.CrossEntropyLoss()          # 对 batch 取平均
loss = loss_fn(logits, y)                # logits: [B, q]，y: [B] 里是类别下标
```

你的 `MLP` 最后一层线性输出的 `logits` 就可以直接送进这个损失。




## 内化加强

理解只是把链条看通了一次。内化是关掉笔记，还能从同一句话自己走到交叉熵和梯度。

只记住这个起点：

> \(q\) 类、样本独立、模型输出概率、最大化 \(P(\mathbf Y\mid\mathbf X)\)。

然后按这个顺序往下写，中间不看笔记：

1. 四个符号：\(\mathbf x^{(i)}\)、\(\mathbf z^{(i)}\)、\(\hat{\mathbf y}^{(i)}\)、\(\mathbf y^{(i)}\)。
2. 独热标签下 \(P(\mathbf y\mid\mathbf x)=\hat y_c\)。
3. 独立 \(\Rightarrow\) 似然是连乘。
4. \(\log\) 把连乘变求和；要最小化就加负号，得到 \(l=-\log\hat y_c\)。
5. 代回 softmax：\(l=-z_c+\log\sum_k e^{z_k}\)。
6. 对 \(z_j\) 求导：\(\partial l/\partial z_j=\hat y_j-y_j\)。

哪一步写不出来，缺的就是那一步，不必整段重读。今天写一遍，明天再写一遍；两次都能从起点走到第 6 步，这条链就算能随时推出来。