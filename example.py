import torch
from scram_pytorch import Scram
import torch.nn as nn

def main():
    p = nn.Parameter(torch.zeros([4], dtype=torch.float32))
    inputs = torch.tensor([[0, 1, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 0]], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    loss_fn = nn.MSELoss()
    
    optimizer = Scram([p], lr=0.5)

    pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
    loss = loss_fn(pred, target)
    
    for step in range(101):
        print(f"step {step}:")
        print(f"p={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = loss_fn(pred, target)

if __name__ == "__main__":
    main()
