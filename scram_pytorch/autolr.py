import random, math
import torch, numpy
from torch.optim.lr_scheduler import LRScheduler

class AutoLR:
    def __init__(
        self,
        optimizer,
        betas = (0.9, 0.99),
        adjustment_rate = 0.1,
    ):
        self.optimizer = optimizer
        self.betas = betas
        self.adjustment_rate = adjustment_rate
        
        self.last_rand_lr = 1
        self.lr_mult = 1
        self.steps = 0
        self.exp_loss = self.exp_loss_sq = 0
        self.exp_lr = self.exp_lr_sq = 0
        self.exp_cov = 0
        self.last_loss = None
        
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        
    def step(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        lr_diff = self.last_rand_lr - self.exp_lr

        if self.last_loss != None:
            loss_delta = loss - self.last_loss
        else:
            loss_delta = 0
        loss_diff = loss_delta - self.exp_loss

        autolr_beta = min(self.steps / (self.steps + 1), self.betas[1])

        self.steps += 1

        self.exp_lr = autolr_beta * self.exp_lr + (1 - autolr_beta) * self.last_rand_lr
        self.exp_lr_sq = autolr_beta * self.exp_lr_sq + (1 - autolr_beta) * self.last_rand_lr ** 2
        self.exp_loss = autolr_beta * self.exp_loss + (1 - autolr_beta) * loss_delta
        self.exp_loss_sq = autolr_beta * self.exp_loss_sq + (1 - autolr_beta) * loss_delta ** 2

        lr_stdev = (self.exp_lr_sq - self.exp_lr ** 2) ** 0.5 or 1
        loss_stdev = (self.exp_loss_sq - self.exp_loss ** 2) ** 0.5 or 1

        cov = (lr_diff / lr_stdev) * (loss_diff / loss_stdev)
        self.exp_cov = autolr_beta * self.exp_cov + (1 - autolr_beta) * cov
        update = self.betas[0] * self.exp_cov + (1 - self.betas[0]) * cov

        self.lr_mult *= math.exp(-self.adjustment_rate * update * self.betas[1])

        #print(f"self.exp_lr={self.exp_lr}, self.exp_loss={self.exp_loss}, self.exp_cov={self.exp_cov}, self.lr_mult={self.lr_mult}")
            
        rand_lr = random.uniform(0.9, 1.1)
        
        self.last_loss = loss
        self.last_rand_lr = rand_lr
        
        self._last_rand_lr = [base_lr * rand_lr * self.lr_mult for base_lr in self.base_lrs]
        
        for i, data in enumerate(zip(self.optimizer.param_groups, self._last_rand_lr)):
            param_group, lr = data
            param_group['lr'] = lr

    def get_last_rand_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_rand_lr

