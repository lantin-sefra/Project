"""
练习 4：train/eval 行为与损失（对应笔记 ch05_5）
先预测再验证；预测错了比预测对更有价值——去笔记里找原因。
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1) Dropout(0.5)，输入全 1
drop = nn.Dropout(0.5)
x = torch.ones(1, 10000)
drop.train()
frac_zero = (drop(x) == 0).float().mean().item()
assert 0.45 < frac_zero < 0.55
nonzero_value = 2                  # 预测：train 模式下，没被丢掉的元素值是多少？
assert drop(x).max().item() == nonzero_value
drop.eval()

assert torch.equal(drop(x), torch.ones(1, 10000))     # 预测：eval 模式的输出等于谁？

# 2) BatchNorm1d 的 running_mean
bn = nn.BatchNorm1d(3)
data = torch.randn(64, 3) * 5 + 2
init_running_mean_is_zero = True      # True / False
assert bool(torch.all(bn.running_mean == 0)) == init_running_mean_is_zero
bn.train()
bn(data)
tracked =1                      # 预测 bn.num_batches_tracked
assert bn.num_batches_tracked.item() == tracked
rm_before = bn.running_mean.clone()
bn.eval()
bn(data)
rm_unchanged = True                   # eval 模式下前向，running_mean 会变吗？True=不变
assert torch.equal(bn.running_mean, rm_before) == rm_unchanged

# 3) CrossEntropy 的内部关系
logits = torch.randn(4, 10)
target = torch.tensor([1, 3, 0, 7])
ce = nn.functional.cross_entropy(logits, target)
# TODO: 用 log_softmax + nll_loss 复现 ce
nll = nn.functional.nll_loss(nn.functional.log_softmax(logits,dim = 1),target,reduction = 'mean')
assert torch.allclose(ce, nll)
# TODO: 写出「错误接法」（先 softmax 再 cross_entropy），并断言它和 ce 不相等
wrong = nn.functional.cross_entropy(nn.functional.softmax(logits,dim = 1), target,reduction = 'mean')
assert not torch.allclose(ce, wrong)
print("ex4 OK")
