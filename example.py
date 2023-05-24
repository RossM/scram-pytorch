import torch
from scram_pytorch import Scram, Simon
import torch.nn as nn

def optimize(inputs, target, optimizer_class):
    p = nn.Parameter(torch.zeros([inputs.shape[1]], dtype=torch.float32))
    loss_fn = nn.MSELoss()
    
    optimizer = optimizer_class([p], lr=0.5, weight_decay=0.1)
    
    for step in range(100):
        optimizer.zero_grad()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

    pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
    loss = loss_fn(pred, target)
    print(f"p={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")

def main():
    inputs = torch.tensor([[0, 1, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 0]], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    print("Original")
    optimize(inputs, target, Simon)
    
    rotation = torch.tensor([[1, -1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 1, -1],
                             [0, 0, 1, 1]], dtype=torch.float32) * (2 ** -0.5)
    inputs = torch.matmul(inputs, rotation)
    print("Rotated")
    optimize(inputs, target, Simon)

if __name__ == "__main__":
    main()
