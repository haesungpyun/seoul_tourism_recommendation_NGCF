import pandas as pd
import torch
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import pickle
import os
from scipy.linalg import get_blas_funcs


class Matrix(nn.Module):
    """
    Manage all operations according to Matrix creation
    """

    def __init__(self, total_df: pd.DataFrame,
                 cols: list,
                 rating_col: str,
                 num_dict: dict,
                 folder_path:str,
                 save_data:bool,
                 device):
        super(Matrix, self).__init__()
        self.df = total_df[cols]
        self.rating_col = rating_col
        self.folder_path = folder_path
        self.save_data = save_data
        self.device = device
        self.n_user = num_dict['user']
        self.n_item = num_dict['item']

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.lap_list = []
        for _ in self.df['year'].unique():
            self.lap_list.append([])

    def create_matrix(self):
        for year in self.df['year'].unique():
            df_tmp = self.df[self.df['year'].isin([year])]
            self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp[self.rating_col]

            # A = [[0, R],[R.T,0]]
            adj_mat = self.adj_mat.tolil()
            R = self.R.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T

            # L = D^-1/2 * A * D^-1/2
            diag = np.count_nonzero(adj_mat.toarray(), axis=1, keepdims=True)
            d_sqrt = np.power(diag, -0.5, dtype=np.float32).squeeze()
            d_sqrt[np.isinf(d_sqrt)] = 0.
            d_mat_inv = np.zeros(adj_mat.shape)

            np.fill_diagonal(d_mat_inv, d_sqrt)
            gemm = get_blas_funcs("gemm", [d_mat_inv, adj_mat.toarray(), d_mat_inv])
            gemm(1, d_mat_inv, adj_mat.toarray(), d_mat_inv)
            adj_mat = sp.dok_matrix(adj_mat)

            year_idx = year % 18
            self.lap_list[year_idx] = self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)

        print('Laplacian Matrix Created!')
        if self.save_data:
            MODEL_PATH = os.path.join(self.folder_path,
                                      f'lap_list' + '.pkl')
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.lap_list, f)
            print('Laplacian  data Saved!')
        return self.lap_list

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = torch.LongTensor(np.mat([coo.row, coo.col]))
        vals = torch.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return torch.sparse.FloatTensor(idxs, vals, coo.shape)


"""
    # for implicit feedback
    def create_matrix(self):
        for user in self.df['userid'].unique():
            df_user = self.df[self.df['userid'].isin([user])]
            for year in df_user['year'].unique():
                df_tmp = df_user[df_user['year'].isin([year])]

                mean = df_tmp[self.rating].quantile(q=0.5)
                df_tmp.loc[df_tmp[self.rating] <= mean, self.rating] = 0
                df_tmp.loc[df_tmp[self.rating] > mean, self.rating] = 1
                self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp[self.rating]
"""
