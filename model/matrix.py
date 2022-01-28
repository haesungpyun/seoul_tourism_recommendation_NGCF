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
                 device):

        self.df = total_df[cols]
        self.device = device
        self.n_user = (total_df['userid'].nunique())
        self.n_item = (total_df['itemid'].nunique())

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.lap_list = torch.empty((total_df['year'].nunique(), self.n_user+self.n_item, self.n_user+self.n_item))

    def create_matrix(self):
        for year in self.df['year'].unique():
            df_tmp = self.df[self.df['year'].isin([year])]
            self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp['congestion_1']

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

            year_idx = year % 18
            self.lap_list[year_idx] = torch.from_numpy(adj_mat.toarray()).to(self.device)
            #self.lap_list[date] = self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)
        print('Laplacian Matrix Created!')
        return self.lap_list


