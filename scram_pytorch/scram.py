import torch
from torch.optim.optimizer import Optimizer

class Scram(Optimizer):
    """SCale and Rotation invAriant Momentum (SCRAM) optimizer"""
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9, 0.99),
        weight_decay: float = 0,
        eps: float = 1e-15,
        limit: float = None,
        add_eps: bool = False,
    ):
        assert lr > 0.
        assert 0. <= betas[0] <= 1.
        assert 0. <= betas[1] <= 1.
        
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            eps = eps,
            limit = limit,
            add_eps = add_eps,
        )
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss =  None
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
                limit = group.get("limit", None)
                add_eps = group.get("add_eps", False)
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    
                exp_avg = state['exp_avg']

                p.data.mul_(1 - lr * wd)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha = 1 - beta1)
                if add_eps:
                  rms = (update ** 2).mean() ** 0.5 + eps
                else:
                  rms = torch.clamp((update ** 2).mean() ** 0.5, min=eps)
                update.div_(rms)
                if limit != None:
                  update = torch.clamp(update, min=-limit, max=limit)
                  
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
        
        return loss
