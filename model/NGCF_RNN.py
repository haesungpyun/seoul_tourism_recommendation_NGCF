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
                 lap_mat: torch.sparse.FloatTensor,
                 eye_mat: torch.sparse.FloatTensor,
                 device):
        super(NGCF_RNN, self).__init__()




