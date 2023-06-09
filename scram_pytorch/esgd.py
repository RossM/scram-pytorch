import torch
from torch.optim.optimizer import Optimizer

class EnsembleSGD(Optimizer):
    """
    Ensemble Stochastic Gradient Descent
    
    This optimizer simulates optimizing a large ensemble of models by maintaining
    two copies of the model weights and randomly selecting one of the copies for
    each scalar weight at each training step.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9, 0.99),
        p = 0.5,
        swap_ratio = 1.0,
        weight_decay: float = 0,
        eps: float = 1e-15,
        ungroup_dims = None,
        normalize = True,
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
            swap_ratio = swap_ratio,
            ungroup_dims = ungroup_dims,
            normalize = normalize,
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
                swap_ratio = group["swap_ratio"]
                ungroup_dims = group.get("ungroup_dims") or []
                normalize = group.get("normalize", True)
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['backup'] = p.data.clone()

                exp_avg = state['exp_avg']
                backup = state['backup']

                if normalize:
                    mean_dims = [x for x in range(0, len(p.data.shape)) if not x in ungroup_dims]
                    grad = grad / ((grad ** 2).mean(dim=mean_dims, keepdim=True) ** 0.5 + eps)

                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)

                p.data.mul_(1 - lr * wd)
                
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1-beta1)
                p.data.add_(update, alpha=-lr)
                
                mask = torch.rand_like(p.data) < update_p
                diff = swap_ratio * mask * (backup - p.data)
                p.data.add_(diff)
                backup.sub_(diff)

        return loss
