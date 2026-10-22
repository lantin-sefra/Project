import torch
import numpy as np
import random

def set_seed(seed : int = 42) ->None :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 下面两行会降低一点性能，换绝对可复现时再开：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed()

    """
    model = Mymodel().to(device)
    for xb, yb in in.loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
    """
    x = torch.arange(12).reshape(3,4)
    print(x)
    sub_idx = torch.tensor([0,2])
    print(x[sub_idx],x[sub_idx].shape)
    print(x.t())

    a = torch.ones(2,3,3)
    b = torch.zeros(2,3,3)
    # c = torch.randn(2,3)
    z1 = torch.cat([a,b],dim = 0)
    print(f"z1 shape:{z1.shape}\n{z1}")

    z = torch.stack([a,b],dim = 2)
    print(f"z shape:\n{z}")

    x1 = torch.ones(1,3)
    x2 = torch.full((1,3),2)
    x3 = torch.randn(1,3)
    x = torch.cat([x1,x2,x3], dim = 0)
    xp = x.permute(1,0)

    print(x)


