import pandas as pd
import torch as t
import scipy.sparse as sp
import numpy as np



class Matrix(object):
    """
    Manage all operations according to Matrix creation
    """
    def __init__(self, total_df:pd.DataFrame,
                 label_col: str,
                 device):

        self.df = total_df
        self.device = device
        self.label_col = label_col
        self.n_user = len(total_df.useridx.unique())
        self.n_item = len(total_df.itemidx.unique())
        self.n_days = len(total_df.date.unique())

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)

        self.lap_list = []
        for _ in range(len(self.df['dateidx'].unique())):
            self.lap_list.append([])

    def create_matrix(self):
        u_i_dict = { k:(v1,v2) for k,v1,v2 in zip(self.df['dateidx'], self.df['useridx'], self.df['itemidx'])}
        for date,(u,i) in u_i_dict.items():
            self.R[u, i] = 1.0

            # A = [[0, R],[R.T,0]]
            adj_mat = self.adj_mat.tolil()
            R = self.R.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T

            # L = D^-1/2 * A * D^-1/2
            diag = np.array(self.adj_mat.sum(1))
            d_sqrt = np.power(diag, -0.5, dtype=np.float32).squeeze()
            d_sqrt[np.isinf(d_sqrt)] = 0.
            d_mat_inv = sp.diags(d_sqrt)
            adj_mat = d_mat_inv.dot(self.adj_mat).dot(d_mat_inv)

            self.lap_list[date] = self._convert_sp_mat_to_sp_tensor(adj_mat).to(device=self.device)

        self.eye_mat = self._convert_sp_mat_to_sp_tensor(sp.eye(self.adj_mat.shape[0])).to(device=self.device)
        return self.lap_list, self.eye_mat


    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = t.LongTensor(np.mat([coo.row, coo.col]))
        vals = t.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return t.sparse.FloatTensor(idxs, vals, coo.shape)


