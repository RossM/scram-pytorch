import torch, argparse
from scram_pytorch import *
from torch.optim import SGD, AdamW
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of using optimizers.")
    parser.add_argument("--optimizer", type=str, default="Scram", help="Optimizer to use (options: Scram, Simon)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Optimizer beta 1")
    parser.add_argument("--beta2", type=float, default=0.99, help="Optimizer beta 2")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.5, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Optimizer weight decay")
    parser.add_argument("--epsilon", type=float, default=1e-15, help="Optimizer epsilon")
    parser.add_argument("--rmsclip", action="store_true", help="Turn on RMS clipping (Simon only)")
    parser.add_argument("--n", type=int, default=1, help="Optimizer n")
    parser.add_argument("--layerwise", action="store_true", help="Layerwise scaling (Simon only)")
    parser.add_argument("--rotate_dimensions", action="store_true", help="Apply a transformation that mixes the model channels while leaving the optimum solution unchanged")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps to perform")
    parser.add_argument("--print_all_steps", action="store_true", help="Print all optimization steps")
    parser.add_argument("--autolr", action="store_true", help="Use autolr")
    
    args = parser.parse_args()
    return args

def optimize(inputs, target, optimizer_class, *, steps=100, print_all_steps=False, autolr=False, opt_args=None):
    p = nn.Parameter(torch.zeros([inputs.shape[1]], dtype=torch.float32))
    
    optimizer = optimizer_class([p], **opt_args)
    if autolr:
        lr_scheduler = AutoLR(optimizer)
    
    for step in range(steps):
        optimizer.zero_grad()
        pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
        loss = ((pred - target) ** 2).mean() + 0.1 * (p ** 2).mean()
        loss.backward()
        if print_all_steps:
            print(f"step={step}\np={p.data}\ngrad={p.grad}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")
        optimizer.step()
        if autolr:
            lr_scheduler.step(loss)
            
    if optimizer_class.__name__ == "AdamWScheduleFree":
        optimizer.eval()

    pred = torch.sigmoid(torch.einsum('y x, x -> y', inputs, p))
    loss = ((pred - target) ** 2).mean() + 0.1 * (p ** 2).mean()
    print(f"step={steps}\np={p.data}\nerr={torch.abs(pred - target).detach()}\nloss={loss}\n")

def main():
    args = parse_args()
    opt_args = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,
        "eps": args.epsilon,
    }
    
    if args.optimizer == "SGD":
        optimizer_class = SGD
        del opt_args["betas"]
        del opt_args["eps"]
        opt_args["momentum"] = args.beta1
    elif args.optimizer == "AdamW":
        optimizer_class = AdamW
    elif args.optimizer == "Scram":
        optimizer_class = Scram
    elif args.optimizer == "Simon":
        optimizer_class = Simon
        opt_args["rmsclip"] = args.rmsclip
        opt_args["layerwise"] = args.layerwise
    elif args.optimizer == "AdamW":
        optimizer_class = torch.optim.AdamW
    elif args.optimizer == "Lion":
        from lion_pytorch import Lion
        optimizer_class = Lion
        del opt_args["eps"]
    elif args.optimizer == "ESGD":
        optimizer_class = EnsembleSGD
        del opt_args["betas"]
    elif args.optimizer == "PowerDescent":
        optimizer_class = PowerDescent
        del opt_args["betas"]
        del opt_args["eps"]
        opt_args["n"] = args.n
    elif args.optimizer == "Mystery":
        optimizer_class = Mystery
        del opt_args["eps"]
    elif args.optimizer == "AdamWScheduleFree":
        from schedulefree import AdamWScheduleFree
        optimizer_class = AdamWScheduleFree
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

    optimize(inputs, target, optimizer_class, steps=args.steps, print_all_steps=args.print_all_steps, autolr=args.autolr, opt_args=opt_args)

if __name__ == "__main__":
    main()
