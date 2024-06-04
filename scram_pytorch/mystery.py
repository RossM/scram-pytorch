import torch
from torch.optim.optimizer import Optimizer

class Mystery(Optimizer):
    """
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.9, 0.99),
        weight_decay: float = 0,
        polyak: bool = False
    ):
        assert lr > 0.
        assert 0. <= betas[0] <= 1.
        assert 0. <= betas[1] <= 1.
        
        self.training = True
        
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
            betas = betas,
            polyak = polyak,
        )
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        self.train()

        loss = None
        if closure != None:
            with torch.enable_grad():
                loss = closure()
    
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                polyak = group["polyak"]
                beta1, beta2 = group["betas"]
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['backup'] = p.data.clone()
                    state['step'] = torch.tensor(0)

                exp_avg = state['exp_avg']
                backup = state['backup']
                step = state['step']
                
                step += 1
                if polyak:
                    beta2 = 1 - 1 / step
                    bias_correction = 1
                else:
                    bias_correction = 1 - beta2 ** step

                grad = grad.sign()

                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)

                p.data.mul_(1 - lr * wd)
                
                p.data.add_(grad, alpha=-lr*(1-beta1))
                p.data.add_(exp_avg, alpha=-lr*beta1/bias_correction)
                backup.add_(exp_avg, alpha=-lr/bias_correction)
                
        return loss

    def train(self, training=True):
        if self.training == training:
            return
        self.training = training
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                backup = state['backup']

                # Swap the values of p and backup, without copying memory
                tmp = p.data
                p.set_(backup.data)
                backup.set_(tmp)
    
    def test(self):
        self.train(False)
        