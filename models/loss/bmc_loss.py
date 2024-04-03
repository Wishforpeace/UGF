import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


import torch


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma,device):
        super(BMCLoss, self).__init__()
        self.device = device
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma),device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var, self.device)


def bmc_loss(pred, target, noise_var,device):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
   
    pred = pred.to(device)
    target = target.to(device)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss