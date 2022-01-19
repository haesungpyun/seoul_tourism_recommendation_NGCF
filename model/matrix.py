import pandas as pd
import torch as t
import scipy.sparse as sp
import numpy as np



class Matrix(object):
    """
    Manage all operations according to Matrix creation
    """
    def __init__(self, users, items, device):
        self.users = users
        self.items = items
        self.device = device
        self.n_user = 672
        self.n_item = 100

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.laplacian_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.sparse_norm_adj = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.eye_mat = sp.dok_matrix((self.sparse_norm_adj.shape[0], self.sparse_norm_adj.shape[0]), dtype=np.float32)

    def create_matrix(self):
        u_m_set = tuple(zip(self.users, self.items))
        for u, m in u_m_set:
            self.R[u-1, m-1] = 1.

        # A = [[0, R],[R.T,0]]
        adj_mat = self.adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_user, self.n_user:] = R
        adj_mat[self.n_user:, :self.n_user] = R.T
        self.adj_mat = adj_mat.todok()

        # L = D^-1/2 * A * D^-1/2
        diag = np.array(self.adj_mat.sum(1))
        d_sqrt = np.power(diag, -0.5, dtype=np.float32).squeeze()
        d_sqrt[np.isinf(d_sqrt)] = 0.
        d_mat_inv = sp.diags(d_sqrt)
        self.laplacian_mat = d_mat_inv.dot(self.adj_mat).dot(d_mat_inv)

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.laplacian_mat).to(device=self.device)
        self.eye_mat = self._convert_sp_mat_to_sp_tensor(sp.eye(self.sparse_norm_adj.shape[0])).to(device=self.device)
        return self.sparse_norm_adj, self.eye_mat

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = t.LongTensor(np.mat([coo.row, coo.col]))
        vals = t.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return t.sparse.FloatTensor(idxs, vals, coo.shape)


