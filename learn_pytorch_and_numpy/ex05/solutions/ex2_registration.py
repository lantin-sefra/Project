import torch.nn as nn


class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]


class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])


assert len(list(Bad().parameters())) == 0
assert len(list(Good().parameters())) == 6

g = Good()
for p in g.layers[0].parameters():
    p.requires_grad = False
trainable = sum(p.numel() for p in g.parameters() if p.requires_grad)
assert trainable == 2 * (10 * 10 + 10), trainable
assert set(g.state_dict().keys()) == {
    f"layers.{i}.{k}" for i in range(3) for k in ("weight", "bias")
}
print("ex2 OK")
