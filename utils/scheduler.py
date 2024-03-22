from torch.optim import SGD,AdamW,lr_scheduler,Adam 
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau




def build_optimizer(args,optimizer_grouped_parameters,epochs):
        if len(args.gpu_ids)>1:
            if args.optim == 'sgd':
                optimizer = SGD(optimizer_grouped_parameters,lr=args.learning_rate,momentum=0.9,weight_decay=1e-4)
            elif args.optim == 'adamw':
                optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
            elif args.optim == "adam":
                optimizer = Adam(optimizer_grouped_parameters,lr=args.learning_rate)
        else:
            if args.optim == 'sgd':
                optimizer = SGD(optimizer_grouped_parameters,lr=args.learning_rate,momentum=0.9,weight_decay=1e-4)
            elif args.optim == 'adamw':
                optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
            elif args.optim == "adam":
                optimizer = Adam(optimizer_grouped_parameters,lr=args.learning_rate)
        # if args.lr_scalar == 'lrstep':
        #     scheduler = lr_scheduler.StepLR(optimizer,args.lr_decay_step,args.lr_decay_ratio)
        # elif args.lr_scalar == 'cosinestep':
        #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs/2,eta_min=1e-6,last_epoch=-1)
        # elif args.lr_scalar == 'cosinestepwarmup':
        #     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        # elif args.lr_scalar == 'onecyclewarmup':
        #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=args.epochs, anneal_strategy='linear')

        scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.8 * epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=0.2 * epochs, after_scheduler=scheduler_steplr)

        # scheduler_warmup = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        return optimizer,scheduler_warmup




class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

