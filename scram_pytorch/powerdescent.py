import torch
from torch.optim.optimizer import Optimizer

class PowerDescent(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9,),
        weight_decay: float = 0,
        eps: float = 1e-15,
        n: int = 1
    ):
        assert lr > 0.
        assert 0. <= betas[0] <= 1.
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
            betas = betas,
            eps = eps,
            n = n
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
                (beta, *_) = group["betas"]
                eps = group["eps"]
                n = group["n"]
                state = self.state[p]

                grad_reshape = grad.reshape((-1, grad.shape[-1]))
                iters = 1
                
                n = min(n, grad.shape[0])
                
                if len(state) == 0:
                    state['a'] = torch.randn((n, grad_reshape.shape[0]), device=grad.device, dtype=grad.dtype)
                    state['sv'] = torch.zeros((n,), device=grad.device, dtype=grad.dtype)
                    iters = 15

                a = state['a']
                sv = state['sv']
                
                for _ in range(iters):
                    b = (a @ grad_reshape).t()
                    b.div_(b.norm(dim=1, keepdim=True) + eps)
                    a = (grad_reshape @ b).t()
                    a.div_(a.norm(dim=1, keepdim=True) + eps)
                    if n > 1:
                        # This applies a linear transformation to make the rows of a orthonormal
                        ch, err = torch.linalg.cholesky_ex(a @ a.t() + eps * torch.eye(n, device=grad.device, dtype=grad.dtype))
                        if err == 0:
                            a = ch.inverse() @ a

                sv.mul_(beta)
                sv.add_((a @ grad_reshape @ b).diag(), alpha=1-beta)
                step = (sv[None, :] * a.t()) @ b.t()

                p.data.mul_(1 - lr * wd)
                p.data.add_(step.reshape(grad.shape), alpha=-lr)
                
                state['a'] = a

        return loss