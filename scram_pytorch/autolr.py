import random, math
import torch, numpy
from torch.optim.lr_scheduler import LRScheduler

class AutoLR:
    """
    A learning rate scheduler that automatically adjusts learning rate by using a zeroth-order
    optimization algorithm to maximize the rate of descent of loss.
    
    The algorithm adds a slight random variation to the learning rate each optimization step.
    It then calculates the covariance between the random value and the change in loss that step,
    and uses the covariance to determine how to adjust learning rate to increase the change
    in loss.
    
    In practice this may underperform a carefully hand-tuned learning rate.
    """
    
    def __init__(
        self,
        optimizer,
        betas = (0.9, 0.99),
        adjustment_rate = 0.1,
        noise_level = 0.1,
    ):
        self.optimizer = optimizer
        self.betas = betas
        self.adjustment_rate = adjustment_rate
        self.noise_level = noise_level
        
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

        # Track exponential moving averages of the 1st and 2nd moments of learning rate and loss delta
        self.exp_lr = autolr_beta * self.exp_lr + (1 - autolr_beta) * self.last_rand_lr
        self.exp_lr_sq = autolr_beta * self.exp_lr_sq + (1 - autolr_beta) * self.last_rand_lr ** 2
        self.exp_loss = autolr_beta * self.exp_loss + (1 - autolr_beta) * loss_delta
        self.exp_loss_sq = autolr_beta * self.exp_loss_sq + (1 - autolr_beta) * loss_delta ** 2

        # Calculate standard deviations of learning rate and loss delta
        lr_stdev = (self.exp_lr_sq - self.exp_lr ** 2) ** 0.5 or 1
        loss_stdev = (self.exp_loss_sq - self.exp_loss ** 2) ** 0.5 or 1

        # Calculate covariance of learning rate and loss delta
        cov = (lr_diff / lr_stdev) * (loss_diff / loss_stdev)
        
        # Track exponential moving average of covariance
        self.exp_cov = autolr_beta * self.exp_cov + (1 - autolr_beta) * cov
        
        # Add a bit of damping to the exponential moving average to avoid periodic oscillation
        update = self.betas[0] * self.exp_cov + (1 - self.betas[0]) * cov

        # Update learning rate based on covariance
        self.lr_mult *= math.exp(-self.adjustment_rate * update * self.betas[1])

        #print(f"self.exp_lr={self.exp_lr}, self.exp_loss={self.exp_loss}, self.exp_cov={self.exp_cov}, self.lr_mult={self.lr_mult}")
            
        # Select a learning rate for the next step
        # We deliberately only track the random part of the learning rate for calculating the adjustment.
        # Including the adjustment in the covariance calculations creates correlations over time that
        # can cause feedback effects and divergence.
        rand_lr = random.uniform(1 - self.noise_level, 1 + self.noise_level)
        
        # Save data for next step
        self.last_loss = loss
        self.last_rand_lr = rand_lr
        
        # Calculate learning rates for each parameter group
        self._last_lr = [base_lr * rand_lr * self.lr_mult for base_lr in self.base_lrs]
        
        # Update optimizer learning rates
        for i, data in enumerate(zip(self.optimizer.param_groups, self._last_lr)):
            param_group, lr = data
            param_group['lr'] = lr

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

