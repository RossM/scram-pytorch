import torch
from torch.optim.optimizer import Optimizer

class Scram(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0,
        epsilon: float = 1e-15,
    ):
        assert lr > 0.
        assert 0. <= beta < 1.
        
        defaults = dict(
            lr = lr,
            beta = beta,
            weight_decay = weight_decay,
            epsilon = epsilon,
        )
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                beta = group["beta"]
                epsilon = group["epsilon"]
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    
                exp_avg = state['exp_avg']
                
                rms = torch.clamp((grad ** 2).mean() ** 0.5, min=epsilon)
                
                exp_avg.mul_(beta).add_(grad, alpha = (1 - beta) / rms)
                p.data.mul_(1 - lr * wd)
                p.add_(exp_avg, alpha=-lr)
