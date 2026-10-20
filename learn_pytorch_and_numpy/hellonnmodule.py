import torch
import torch.nn as nn
import torch.nn.functional as F 

torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self,
        in_dim = 784, 
        hidden = 128, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,hidden)
        self.fc2 = nn.Linear(hidden,num_classes)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
x = torch.randn(4,784)
logits = model(x)
print(logits.shape)


n = sum(p.numel() for p in model.parameters())
print(n)

for name, p in model.named_parameters():
    print(name,tuple(p.shape))

sd = model.state_dict()
print(list(sd.keys()))