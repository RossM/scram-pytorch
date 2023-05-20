import torch
from scram_pytorch import Scram
import torch.nn as nn

def optimize(inputs, target):
    p = nn.Parameter(torch.zeros([inputs.shape[1]], dtype=torch.float32))
    loss_fn = nn.MSELoss()
    
    optimizer = Scram([p], lr=0.5)

    pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
    loss = loss_fn(pred, target)
    
    for step in range(11):
        print(f"step {step}:")
        print(f"p={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = loss_fn(pred, target)

def main():
    inputs = torch.tensor([[0, 1, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 0]], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    print("Original")
    optimize(inputs, target)
    
    rotation = torch.tensor([[1, -1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 1, -1],
                             [0, 0, 1, 1]], dtype=torch.float32) * (2 ** -0.5)
    inputs = torch.matmul(inputs, rotation)
    print("Rotated")
    optimize(inputs, target)

if __name__ == "__main__":
    main()
