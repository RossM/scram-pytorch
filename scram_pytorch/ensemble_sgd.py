import torch
from torch.optim.optimizer import Optimizer

class EnsembleSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0,
        eps: float = 1e-15,
    ):
        assert lr > 0.
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
            eps = eps,
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
                eps = group["eps"]
                state = self.state[p]
                
                if len(state) == 0:
                    state['backup'] = p.data.clone()
                    
                backup = state['backup']

                grad.div_((grad ** 2).mean() ** 0.5 + eps)

                p.data.mul_(1 - lr * wd)
                
                p.data.add_(grad, alpha=-lr)
                
                mask = torch.rand_like(p.data) < 0.5
                diff = mask * (backup - p.data)
                p.data.add_(diff)
                backup.sub_(diff)

        return loss
