import torch
import torch.nn as nn

torch.manual_seed(0)

# 1) Dropout
drop = nn.Dropout(0.5)
x = torch.ones(1, 10000)
drop.train()
frac_zero = (drop(x) == 0).float().mean().item()
assert 0.45 < frac_zero < 0.55
nonzero_value = drop(x).max().item()
assert nonzero_value == 2.0
drop.eval()
assert torch.equal(drop(x), x)

# 2) BatchNorm 的 running_mean
bn = nn.BatchNorm1d(3)
data = torch.randn(64, 3) * 5 + 2
assert torch.all(bn.running_mean == 0)
bn.train()
bn(data)
assert torch.all(bn.running_mean != 0)
assert bn.num_batches_tracked.item() == 1
rm_before = bn.running_mean.clone()
bn.eval()
bn(data)
assert torch.equal(bn.running_mean, rm_before)

# 3) CrossEntropy = NLL(LogSoftmax)
logits = torch.randn(4, 10)
t = torch.tensor([1, 3, 0, 7])
ce = nn.functional.cross_entropy(logits, t)
nll = nn.functional.nll_loss(torch.log_softmax(logits, dim=1), t)
assert torch.allclose(ce, nll)
wrong = nn.functional.cross_entropy(torch.softmax(logits, dim=1), t)
assert not torch.allclose(ce, wrong)
print("ex4 OK")
