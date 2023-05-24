import torch
from torch.optim.optimizer import Optimizer

class Simon(Optimizer):
    """SIgma MOmeNtum (SIMON) optimizer
    
    This optimizer uses the standard deviation of each gradient over time, computed
    using an exponential moving average, as an estimate of the second derivative of
    the loss in order to take small steps when loss is changing quickly and large
    steps when it is changing smoothly."""
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9, 0.99),
        weight_decay: float = 0,
        eps: float = 1e-15,
        rmsclip: bool = False,
        **kwargs,
    ):
        assert lr > 0.
        assert 0. <= betas[0] <= 1.
        assert 0. <= betas[1] <= 1.
        
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            eps = eps,
            rmsclip = rmsclip,
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
                rmsclip = group["rmsclip"]
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                p.data.mul_(1 - lr * wd)
                
                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)
                exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1-beta2)
                
                stdev = (exp_avg_sq - exp_avg ** 2) ** 0.5
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1-beta1)
                if rmsclip:
                    rms = (update ** 2).mean() ** 0.5
                    stdev = torch.max(stdev, rms)
                update = update / (stdev + eps)
                # The factor of beta2 corrects stdev from a population standard deviation to a
                # sample standard deviation
                p.add_(update, alpha=-lr * beta2)
        
        return loss