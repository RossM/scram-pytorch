import torch, argparse
from scram_pytorch import Scram, Simon
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of using optimizers.")
    parser.add_argument("--optimizer", type=str, default="Scram", help="Optimizer to use (options: Scram, Simon)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Optimizer beta 1")
    parser.add_argument("--beta2", type=float, default=0.99, help="Optimizer beta 2")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.5, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Optimizer weight decay")
    parser.add_argument("--epsilon", type=float, default=1e-15, help="Optimizer epsilon")
    parser.add_argument("--rmsclip", action="store_true", help="Rurn on RMS clipping (Simon only)")
    parser.add_argument("--rotate_dimensions", action="store_true", help="Apply a transformation that mixes the model channels while leaving the optimum solution unchanged")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps to perform")
    parser.add_argument("--print_all_steps", action="store_true", help="Print all optimization steps")
    
    args = parser.parse_args()
    return args

def optimize(inputs, target, optimizer_class, *, steps=100, print_all_steps=False, opt_args=None):
    p = nn.Parameter(torch.zeros([inputs.shape[1]], dtype=torch.float32))
    loss_fn = nn.MSELoss()
    
    optimizer = optimizer_class([p], **opt_args)
    
    for step in range(steps):
        optimizer.zero_grad()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = loss_fn(pred, target)
        if print_all_steps:
            print(f"step={step}\np={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")
        loss.backward()
        optimizer.step()

    pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
    loss = loss_fn(pred, target)
    print(f"step={steps}\np={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")

def main():
    args = parse_args()
    opt_args = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,
        "epsilon": args.epsilon,
        "rmsclip": args.rmsclip,
    }
    
    if args.optimizer == "Scram":
        optimizer_class = Scram
    elif args.optimizer == "Simon":
        optimizer_class = Simon
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    inputs = torch.tensor([[0, 1, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 0]], dtype=torch.float32)
    target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)

    if args.rotate_dimensions:
        rotation = torch.tensor([[1, -1, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 1, -1],
                                 [0, 0, 1, 1]], dtype=torch.float32) * (2 ** -0.5)
        inputs = torch.matmul(inputs, rotation)

    optimize(inputs, target, optimizer_class, steps=args.steps, print_all_steps=args.print_all_steps, opt_args=opt_args)

if __name__ == "__main__":
    main()
