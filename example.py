import torch
from scram_pytorch import Scram
import torch.nn as nn

def main():
    p = nn.Parameter(torch.zeros([4], dtype=float))
    inputs = torch.tensor([[0, 1, 1, 0],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 1]], dtype=float)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=float)
    loss_fn = nn.MSELoss()
    
    optimizer = Scram([p], lr=0.5, betas=(0.9,0.9))
    
    for step in range(100):
        optimizer.zero_grad()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        print(f"step {step}:")
        print(f"p={p.data}\npred={pred}\ntarget={target}\nloss={loss}")

if __name__ == "__main__":
    main()
