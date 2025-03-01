import torch
from torch.optim.lr_scheduler import _LRScheduler
from warmup_scheduler import GradualWarmupScheduler
from omegaconf import DictConfig


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    Gradually warm-up (increasing) the learning rate in the optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier (float): Target learning rate = base_lr * multiplier.
        total_epoch (int): Number of epochs over which to warm up.
        after_scheduler (_LRScheduler, optional): Scheduler to use after warmup.
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super().__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * (((self.multiplier - 1.) * self.last_epoch / self.total_epoch) + 1.)
                    for base_lr in self.base_lrs]


def get_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer) -> _LRScheduler:
    """
    Create a learning rate scheduler using cosine annealing after a gradual warmup.

    Args:
        cfg (DictConfig): Configuration object containing scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.

    Returns:
        _LRScheduler: A learning rate scheduler instance.
    """
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler_t_max, eta_min=cfg.min_lr
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=cfg.warmup_epochs, after_scheduler=scheduler_cosine
    )
    return scheduler
