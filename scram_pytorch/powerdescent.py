import torch
from torch.optim.optimizer import Optimizer

class PowerDescent(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0,
        eps: float = 1e-15,
        n: int = 1
    ):
        assert lr > 0.
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
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
                eps = group["eps"]
                n = group["n"]
                state = self.state[p]

                grad_reshape = grad.reshape((-1, grad.shape[-1]))
                iters = 1
                
                if len(state) == 0:
                    state['a'] = torch.randn((n, grad_reshape.shape[0]), device=grad.device, dtype=grad.dtype)
                    iters = 15

                a = state['a']
                
                for _ in range(iters):
                    b = (a @ grad_reshape).t()
                    b.div_(b.norm() + eps)
                    a = (grad_reshape @ b).t()
                    if n > 1:
                        # This applies a linear transformation to make the rows of a orthonormal
                        ch, err = torch.linalg.cholesky_ex(a @ a.t() + eps * torch.eye(n, device=grad.device, dtype=grad.dtype))
                        if err == 0:
                            a = ch.inverse() @ a
                    a.div_(a.norm() + eps)

                sv = (a @ grad_reshape @ b).diag()[None, :]
                step = (sv * a.t()) @ b.t()

                p.data.mul_(1 - lr * wd)
                p.data.add_(step.reshape(grad.shape), alpha=-lr)
                
                state['a'] = a

        return loss