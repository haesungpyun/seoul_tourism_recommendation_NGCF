import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class NGCF(nn.Module):
    def __init__(self,
                 embed_size: int,
                 layer_size: list,
                 node_dropout: float,
                 mess_dropout: list,
                 mlp_ratio: float,
                 lap_list: list,
                 num_dict: dict,
                 batch_size: int,
                 device):
        super(NGCF, self).__init__()

        self.n_user = num_dict['user']
        self.n_item = num_dict['item']

        self.emb_size = embed_size
        self.weight_size = layer_size
        self.n_layer = len(self.weight_size)
        self.batch_size = batch_size

        self.device = device

        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.mlp_ratio = mlp_ratio

        # self.user_embedding = nn.Parameter(torch.randn(self.n_user, self.emb_size))
        # self.item_embedding = nn.Parameter(torch.randn(self.n_item, self.emb_size))
        self.date_emb = nn.Embedding(num_dict['date'], self.emb_size)
        self.day_emb = nn.Embedding(num_dict['day'], self.emb_size)
        self.sex_emb = nn.Embedding(num_dict['sex'], self.emb_size)
        self.age_emb = nn.Embedding(num_dict['age'], self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)

        self.user_lin = []
        self.lin_1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size // 2, bias=True)
        self.lin_2 = nn.Linear(in_features=self.emb_size // 2, out_features=self.emb_size)
        self.user_lin.append(self.lin_1)
        self.user_lin.append(nn.LeakyReLU())
        self.user_lin.append(self.lin_2)
        self.user_lin.append(nn.LeakyReLU())
        self.user_lin = nn.Sequential(*self.user_lin)


        self.w1_list = []
        self.w2_list = []
        self.node_dropout_list = []
        self.mess_dropout_list = []

        self.lap_list = lap_list

        self.set_layers()

    def set_layers(self):

        initializer = nn.init.xavier_uniform_

        # initial embedding layer
        initializer(self.user_embedding.weight)
        initializer(self.item_embedding.weight)

        initializer(self.day_emb.weight)
        initializer(self.sex_emb.weight)
        initializer(self.age_emb.weight)
        initializer(self.date_emb.weight)

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

    def sparse_dropout(self, L, dateid):
        lap = torch.FloatTensor(torch.empty(*L.size())).to(self.device)
        for i in dateid:
            mat = sp.dok_matrix(L[i].cpu().detach().numpy())
            mat = self._convert_sp_mat_to_sp_tensor(mat)
            node_mask = nn.Dropout(self.node_dropout)(torch.tensor(np.ones(mat._nnz()))).type(torch.bool)
            i = mat._indices()
            v = mat._values()
            i = i[:, node_mask]
            v = v[node_mask]
            lap[i] = torch.sparse.FloatTensor(i, v, mat.shape).to_dense().to(self.device)
        return lap

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = torch.LongTensor(np.mat([coo.row, coo.col]))
        vals = torch.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return torch.sparse.FloatTensor(idxs, vals, coo.shape)

    def forward(self, year, u_id, age, day, sex, pos_item, neg_item, node_flag):
        age = self.age_emb(age)
        date = self.date_emb(day)
        sex = self.sex_emb(sex)
        feats = torch.cat((age, date, sex), dim=0)
        user_mlp = self.user_lin(feats)

        year_idx = year.unique()[0] % 18
        L = self.lap_list[year_idx].to(self.device)

        with torch.no_grad():
            self.user_embedding.weight[u_id] = \
                self.user_embedding.weight[u_id] * (1 - self.mlp_ratio) + user_mlp * self.mlp_ratio

        E = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_E = [E]

        for i in range(self.n_layer):
            if node_flag:
                # node dropout laplacian matrix
                L = self.sparse_dropout(L, year)
            else:
                L = L

            L_E = torch.mm(L, E)
            L_E_W1 = self.w1_list[i](L_E)

            E_W1 = self.w1_list[i](E)

            L_E_E = (L_E * E)
            L_E_E_W2 = self.w2_list[i](L_E_E)

            message_embedding = L_E_W1 + E_W1 + L_E_E_W2

            E = nn.LeakyReLU(negative_slope=0.2)(message_embedding)

            E = self.mess_dropout_list[i](E)

            norm_embedding = F.normalize(E, p=2, dim=1)

            all_E += [norm_embedding]
        all_E = torch.cat(all_E, dim=2)
        self.all_users_emb = all_E[:, :self.n_user, :]
        self.all_items_emb = all_E[:, self.n_user:, :]

        u_embeddings = self.all_users_emb[:, u_id, :]
        pos_i_embeddings = self.all_items_emb[:, pos_item, :]
        neg_i_embeddings = torch.empty(0)
        if len(neg_item) > 0:
            neg_i_embeddings = self.all_items_emb[:, neg_item, :]
        return u_embeddings, pos_i_embeddings, neg_i_embeddings