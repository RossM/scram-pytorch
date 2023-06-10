import torch
from torch.optim.optimizer import Optimizer

class Simon(Optimizer):
    """
    SIgma MOmeNtum (SIMON) optimizer
    
    This optimizer uses the standard deviation of each gradient over time, computed
    using an exponential moving average, as an estimate of the second derivative of
    the loss in order to take small steps when loss is changing quickly and large
    steps when it is changing smoothly.
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        betas: coefficients used for computing a running average of gradient and curvature,
            and for damping oscillations caused by momentum
        weight_decay: decoupled way decay coefficient
        eps: term added to the denominator to improve numerical stability
        rmsclip: whether to limit the size of updatea when curvature is very small
        layerwise: whether to calculate curvature per-parameter rather than per-weight.
            typically causes slower convergence, but halves memory usage.
        normalize: whether to normalize gradients per-parameter before other calculations
        distance_weighted: whether to use distance weighted running averages. may give
            a more accurate estimation of curvature, but still experimental.
    """
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
        distance_weighted: bool = False,
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
            distance_weighted = distance_weighted,
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
                rmsclip = group.get("rmsclip", False)
                layerwise = group.get("layerwise", False)
                normalize = group.get("normalize", False)
                distance_weighted = group.get("distance_weighted", False)
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    if layerwise:
                        state['exp_avg_sq'] = torch.zeros([], device=p.device, dtype=p.dtype)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    if distance_weighted:
                        state['exp_distance'] = torch.ones([], device=p.device, dtype=p.dtype)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                if normalize:
                    grad = grad / max((grad ** 2).mean() ** 0.5, eps)

                p.data.mul_(1 - lr * wd)
                
                if distance_weighted:
                    exp_distance = state['exp_distance']
                    distance = grad.abs()
                    weight = distance / (exp_distance + distance)
                    exp_distance.mul_(beta2).add(distance, alpha=1-beta2)
                    exp_avg.mul_(1 - weight).add_(grad * weight)
                else:
                    exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)
                
                if layerwise:
                    exp_avg_sq.mul_(beta2).add_((grad**2).mean(), alpha=1-beta2)
                    var = exp_avg_sq - (exp_avg ** 2).mean()
                else:
                    exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1-beta2)
                    var = exp_avg_sq - exp_avg ** 2
                
                stdev = torch.clamp(var, min=0) ** 0.5

                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1-beta1)

                if rmsclip:
                    rms = (update ** 2).mean() ** 0.5
                    stdev = torch.max(stdev, rms)

                update = update / (stdev + eps)

                # The factor of beta2 corrects stdev from a population standard deviation to a
                # sample standard deviation
                p.add_(update, alpha=-lr * beta2)

        return loss
