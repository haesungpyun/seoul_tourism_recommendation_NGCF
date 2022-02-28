import pandas as pd
import torch
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import pickle
import os
from datetime import datetime
from parsers import args


class Matrix(nn.Module):
    """
    Manage all operations according to Matrix creation
    """
    def __init__(self, total_df: pd.DataFrame,
                 cols: list,
                 rating_col: str,
                 num_dict: dict,
                 folder_path: str,
                 save_data: bool,
                 device):
        super(Matrix, self).__init__()
        self.df = total_df[cols]
        self.rating_col = rating_col
        self.folder_path = folder_path
        self.save_data = save_data
        self.device = device
        self.n_user = num_dict['user']
        self.n_item = num_dict['item']

        # R: user 수 x item 수 matrix, user가 평가한 item의 rating으로 이루어진 matrix
        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        # adj_mat: user-user, item-item, user-item간 연결성을 파악할 수 있도록 모두 모아둔 matrix
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        # 년도 별로 user의 선호도(visitor)가 다르고, user가 소비한 item의 종류도 달라 연도를 구분하여 각 년도에 맞는 matrix 생성
        self.lap_list = []
        for _ in self.df['year'].unique():
            self.lap_list.append([])

    def create_matrix(self):
        for year in self.df['year'].unique():
            # 각 연도 별, user가 평가한 item의 선호도(visitor)를 R에 저장 (user 별 일정 기준 이하:0, 이상: explicit rating)
            df_tmp = self.df[self.df['year'].isin([year])]
            self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp[self.rating_col]

            # A(adj_mat) = [[0, R],
            #              [R.T,0]]
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

            # laplacian matirx 생성
            adj_mat = np.linalg.multi_dot([d_mat_inv, adj_mat.toarray(), d_mat_inv])
            adj_mat = sp.dok_matrix(adj_mat)

            # 연도 별로 저장 18년 : 0, 19년 : 1
            year_idx = year % 18
            self.lap_list[year_idx] = self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)

        print('Laplacian Matrix Created!')
        if self.save_data:
            d1 = datetime.now()
            PATH = os.path.join(self.folder_path,f'lap_list_implicit_{args.epoch}_{args.batch_size}_{args.lr}_{args.emb_ratio}_{args.scaler}_{d1.month}_{d1.day}_{d1.hour}_{d1.minute}' + '.pkl')
            with open(PATH, 'wb') as f:
                pickle.dump(self.lap_list, f)
            print('Laplacian data Saved!')
        return self.lap_list

    # sp_matrix를 tensor로 변형하여 넘겨줌
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

                mean = df_tmp[self.rating_col].quantile(q=0.25)
                df_tmp.loc[df_tmp[self.rating_col] <= mean, self.rating_col] = 0
                df_tmp.loc[df_tmp[self.rating_col] > mean, self.rating_col] = 1
                self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp[self.rating_col]
"""
"""
    # for explicit feedback
    def create_matrix(self):
        for year in self.df['year'].unique():
            df_tmp = self.df[self.df['year'].isin([year])]
            self.R[df_tmp['userid'], df_tmp['itemid']] = df_tmp[self.rating_col]
"""
