import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NGCF(nn.Module):
    def __init__(self,
                 n_user: int,
                 n_item: int,
                 embed_size: int,
                 layer_size: list,
                 node_dropout: float,
                 mess_dropout: list,
                 lap_mat: t.sparse.FloatTensor,
                 eye_mat: t.sparse.FloatTensor,
                 device):
        super(NGCF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item

        self.emb_size = embed_size
        self.weight_size = layer_size
        self.n_layer = len(self.weight_size)

        self.device = device

        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        # self.user_embedding = nn.Parameter(t.randn(self.n_user, self.emb_size))
        # self.item_embedding = nn.Parameter(t.randn(self.n_item, self.emb_size))
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)

        self.user_lin = []
        self.lin_1 = nn.Linear(in_features=4, out_features=self.emb_size // 2, bias=True)
        self.lin_2 = nn.Linear(in_features=self.emb_size // 2, out_features=self.emb_size, bias=True)
        self.user_lin.append(self.lin_1)
        self.user_lin.append(self.lin_2)
        self.user_lin = nn.Sequential(*self.layer_list)

        self.w1_list = []
        self.w2_list = []
        self.node_dropout_list = []
        self.mess_dropout_list = []

        self.L = lap_mat
        self.eye_mat = eye_mat

        self.set_layers()

    def set_layers(self):

        initializer = nn.init.xavier_uniform_

        # initial embedding layer
        initializer(self.item_embedding.weight)
        initializer(self.user_embedding.weight)

        weight_size_list = [self.emb_size] + self.weight_size

        # set propagation layer
        for k in range(self.n_layer):
            # set W1, W2 as Linear layer
            self.w1_list.append(
                nn.Linear(in_features=weight_size_list[k], out_features=weight_size_list[k + 1], bias=True))
            self.w2_list.append(
                nn.Linear(in_features=weight_size_list[k], out_features=weight_size_list[k + 1], bias=True))

            # node_dropout on Laplacian matrix
            if self.node_dropout is not None:
                self.node_dropout_list.append(nn.Dropout(p=self.node_dropout))

            # mess_dropout on l-th message
            if self.mess_dropout is not None:
                self.mess_dropout_list.append(nn.Dropout(p=self.mess_dropout[k]))

        self.w1_list = nn.Sequential(*self.w1_list)
        self.w2_list = nn.Sequential(*self.w2_list)
        self.node_dropout_list = nn.Sequential(*self.node_dropout_list)
        self.mess_dropout_list = nn.Sequential(*self.mess_dropout_list)

    def sparse_dropout(self, mat):
        node_mask = nn.Dropout(self.node_dropout)(t.tensor(np.ones(mat._nnz()))).type(t.bool)
        i = mat._indices()
        v = mat._values()
        i = i[:, node_mask]
        v = v[node_mask]

        drop_mat = t.sparse.FloatTensor(i, v, mat.shape).to(self.device)
        return drop_mat

    def forward(self, user_features, pos_features, neg_features, node_flag, mlp_ratio):
        user_idx = user_features[0]
        user_features = user_features[1:]
        pos_idx = pos_features[0]
        neg_idx = neg_features[0]

        user_mlp = self.user_lin(user_features)

        with t.no_grad():
            self.user_embedding.weight[user_idx] = self.user_embedding.weight[user_idx] * (1-mlp_ratio) + user_mlp * mlp_ratio

        E = t.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_E = [E]

        for i in range(self.n_layer):
            if node_flag:
                # node dropout laplacian matrix
                L = self.sparse_dropout(self.L)
            else:
                L = self.L

            L_I_E = t.mm(L + self.eye_mat, E)
            L_I_E_W1 = self.w1_list[i](L_I_E)

            L_E = t.mm(L, E)
            L_E_E = L_E * E
            L_I_E_W2 = self.w2_list[i](L_E_E)

            message_embedding = L_I_E_W1 + L_I_E_W2

            E = nn.LeakyReLU(negative_slope=0.2)(message_embedding)

            E = self.mess_dropout_list[i](E)

            norm_embedding = F.normalize(E, p=2, dim=1)

            all_E += [norm_embedding]

        all_E = t.cat(all_E, dim=1)
        self.all_users_emb = all_E[:self.n_user, :]
        self.all_items_emb = all_E[self.n_user:, :]

        u_embeddings = self.all_users_emb[user_idx, :]
        pos_i_embeddings = self.all_items_emb[pos_idx, :]
        neg_i_embeddings = t.empty(0)
        if len(neg_idx) > 0:
            neg_i_embeddings = self.all_items_emb[neg_idx, :]
        return u_embeddings, pos_i_embeddings, neg_i_embeddings



