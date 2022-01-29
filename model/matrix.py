import pandas as pd
import torch
import scipy.sparse as sp
import numpy as np


class Matrix(object):
    """
    Manage all operations according to Matrix creation
    """
    def __init__(self, total_df:pd.DataFrame,
                 cols: list,
                 num_dict:dict,
                 device):

        self.df = total_df[cols]
        self.device = device
        self.n_user = num_dict['user']
        self.n_item = num_dict['item']

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.lap_list = torch.empty((total_df['year'].nunique(), self.n_user+self.n_item, self.n_user+self.n_item))

    def create_matrix(self):
        for year in self.df['year'].unique():
            df_tmp = self.df[self.df['year'].isin([year])]
            print(self.R.shape)
            print(df_tmp['userid'].max())
            print(df_tmp['itemid'].max())
            self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp['congestion_1']

            # A = [[0, R],[R.T,0]]
            adj_mat = self.adj_mat.tolil()
            R = self.R.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T

            # L = D^-1/2 * A * D^-1/2
            diag = np.count_nonzero(adj_mat.toarray(), axis=1, keepdims=True)
            d_sqrt = np.power(diag, -0.5, dtype=np.float32).squeeze()
            d_sqrt[np.isinf(d_sqrt)] = 0.
            d_mat_inv = sp.diags(d_sqrt)
            adj_mat = d_mat_inv.dot(adj_mat.toarray()).dot(d_mat_inv)

            year_idx = year % 18
            #self.lap_list[year_idx] = torch.from_numpy(self.adj_mat).to(self.device)
            self.lap_list[year_idx] = self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)
        print('Laplacian Matrix Created!')
        return self.lap_list

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = torch.LongTensor(np.mat([coo.row, coo.col]))
        vals = torch.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return torch.sparse.FloatTensor(idxs, vals, coo.shape)
