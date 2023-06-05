import torch
from torch.optim.optimizer import Optimizer

class EnsembleSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9, 0.99),
        p = 0.5,
        weight_decay: float = 0,
        eps: float = 1e-15,
    ):
        assert lr > 0.
        assert 0. <= betas[0] <= 1.
        assert 0. <= betas[1] <= 1.
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
            betas = betas,
            eps = eps,
            p = p,
        )
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure != None:
            with torch.enable_grad():
                loss = closure()
    
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                update_p = group["p"]
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['backup'] = p.data.clone()

                exp_avg = state['exp_avg']
                backup = state['backup']

                grad.div_((grad ** 2).mean() ** 0.5 + eps)

                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)

                p.data.mul_(1 - lr * wd)
                
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1-beta1)
                p.data.add_(update, alpha=-lr)
                
                mask = torch.rand_like(p.data) < update_p
                diff = mask * (backup - p.data)
                p.data.add_(diff)
                backup.sub_(diff)

        return loss
