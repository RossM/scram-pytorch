import random
import torch, numpy
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
        layerwise: bool = False,
        normalize: bool = False,
        autolr: bool = False,
        autolr_beta: float = 0.99,
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
            layerwise = layerwise,
            normalize = normalize,
        )
        
        self.autolr = autolr
        self.autolr_beta = autolr_beta
        self.last_lr = None
        self.lr_mult = 1
        self.autolr_steps = 0
        self.exp_loss = self.exp_loss_sq = 0
        self.exp_lr = self.exp_lr_sq = 0
        self.exp_cov = 0
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure != None:
            with torch.enable_grad():
                loss = closure()
    
        if self.autolr and loss != None:
            if self.last_lr != None:
                if loss.__class__ == torch.Tensor:
                    loss = loss.item()
                loss_delta = loss - self.last_loss
                lr_diff = self.last_lr - self.exp_lr
                loss_diff = loss_delta - self.exp_loss
                autolr_beta = min(self.autolr_steps / (self.autolr_steps+ 1), self.autolr_beta)
                self.autolr_steps += 1
                self.exp_lr = autolr_beta * self.exp_lr + (1 - autolr_beta) * self.last_lr
                self.exp_lr_sq = autolr_beta * self.exp_lr_sq + (1 - autolr_beta) * self.last_lr ** 2
                self.exp_loss = autolr_beta * self.exp_loss + (1 - autolr_beta) * loss_delta
                self.exp_loss_sq = autolr_beta * self.exp_loss_sq + (1 - autolr_beta) * loss_delta ** 2
                lr_stdev = (self.exp_lr_sq - self.exp_lr ** 2) ** 0.5 or 1
                loss_stdev = (self.exp_loss_sq - self.exp_loss ** 2) ** 0.5 or 1
                cov = (lr_diff / lr_stdev) * (loss_diff / loss_stdev)
                self.exp_cov = autolr_beta * self.exp_cov + (1 - autolr_beta) * cov                
                self.lr_mult *= 1.1 ** -(self.exp_cov * autolr_beta)
                #print(f"self.exp_lr={self.exp_lr}, self.exp_loss={self.exp_loss}, self.exp_cov={self.exp_cov}, self.lr_mult={self.lr_mult}")
                
            cur_lr = random.uniform(0.9, 1.1)
            
            self.last_loss = loss
            self.last_lr = cur_lr
            
            cur_lr *= self.lr_mult
        else:
            cur_lr = 1
            
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                rmsclip = group.get("rmsclip", False)
                layerwise = group.get("layerwise", False)
                normalize = group.get("normalize", False)
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                if normalize:
                    grad = grad / max((grad ** 2).mean() ** 0.5, eps)

                p.data.mul_(1 - lr * wd)
                
                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)
                exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1-beta2)
                
                if layerwise:
                    stdev = (exp_avg_sq - exp_avg ** 2).mean() ** 0.5
                else:
                    stdev = (exp_avg_sq - exp_avg ** 2) ** 0.5
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1-beta1)
                if rmsclip:
                    rms = (update ** 2).mean() ** 0.5
                    stdev = torch.max(stdev, rms)
                update = update / (stdev + eps)
                # The factor of beta2 corrects stdev from a population standard deviation to a
                # sample standard deviation
                p.add_(update, alpha=-lr * beta2 * cur_lr)
        
        return loss
