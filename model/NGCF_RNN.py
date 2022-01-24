import torch
import torch.nn as nn
import numpy as np
from NGCF import NGCF


class NGCF_RNN(nn.Module):
    def __init__(self,
                 n_user: int,
                 n_item: int,
                 embed_size: int,
                 layer_size: list,
                 node_dropout: float,
                 mess_dropout: list,
                 lap_list: torch.sparse.FloatTensor,
                 eye_mat: torch.sparse.FloatTensor,
                 device):
        super(NGCF_RNN, self).__init__()

        NGCF = NGCF(n_user=n_user,
                     n_item=n_item,
                     embed_size=64,
                     layer_size=[64, 64, 64],
                     node_dropout=0.2,
                     mess_dropout=[0.1, 0.1, 0.1],
                     mlp_ratio=0.5,
                     lap_list=lap_list,
                     eye_mat=eye_mat,
                     device=device).to(device=device)



