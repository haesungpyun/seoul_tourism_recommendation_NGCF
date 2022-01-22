import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, weight_decay, batch_size):
        super(BPR, self).__init__()
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def forward(self, u_idx, pos_idx, neg_idx):
        x_upos = torch.mul(u_idx, pos_idx).sum(dim=1)
        x_uneg = torch.mul(u_idx, neg_idx).sum(dim=1)
        x_upn = x_upos - x_uneg
        log_prob = F.logsigmoid(x_upn).sum()
        regularization = self.weight_decay * (u_idx.norm(dim=1).pow(2).sum() +
                                              pos_idx.norm(dim=1).pow(2).sum() + neg_idx.norm(dim=1).pow(2).sum())
        return (-log_prob + regularization) / self.batch_size
