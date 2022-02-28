import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

# 기존 bpr loss의 경우 x_upn = ,,, 식에 torch.abs가 없음. implicit하게 rating을 사용하였기 때문.
# 해당 모델 구현 시 explicit한 visitor의 수를 일정 기준으로 0으로 만들어 implicit하게 사용하였기 때문에
# 기존처럼 abs가 없다면 양수 - 음수가 되어 loss 계산, 역전파에 이상이 생김.
class BPR(nn.Module):
    def __init__(self, weight_decay, batch_size):
        super(BPR, self).__init__()
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def forward(self, u_idx, pos_idx, neg_idx):
        x_upos = torch.mul(u_idx, pos_idx).sum(dim=1)
        x_uneg = torch.mul(u_idx, neg_idx).sum(dim=1)
        x_upn = torch.abs(x_upos) - torch.abs(x_uneg)
        log_prob = F.logsigmoid(x_upn).sum()
        regularization = self.weight_decay * (LA.norm(u_idx, dim=1).pow(2).sum() +
                                              LA.norm(pos_idx, dim=1).pow(2).sum() + LA.norm(neg_idx, dim=1).pow(2).sum())
        return (-log_prob + regularization) / self.batch_size
