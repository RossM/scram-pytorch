import torch
from torch.optim.optimizer import Optimizer

class PowerDescent(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0,
    ):
        assert lr > 0.
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
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
                state = self.state[p]

                grad_reshape = grad.reshape((-1, grad.shape[-1]))
                iters = 1
                
                if len(state) == 0:
                    state['a'] = torch.randn((1, grad_reshape.shape[0]), device=grad.device, dtype=grad.dtype)
                    iters = 15

                a = state['a']
                
                for _ in range(iters):
                    b = (a @ grad_reshape).t()
                    b.div_(b.norm())
                    a = (grad_reshape @ b).t()
                    a.div_(a.norm())

                step = -lr * (a @ grad_reshape @ b).squeeze()

                p.data.add_((a * b).t().reshape(grad.shape), alpha = step)
                
                state['a'] = a

        return loss