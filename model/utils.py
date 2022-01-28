import random

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def split_train_test(root_dir:str,
                     train_by_destination:bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    pick each unique userid row, and add to the testset, delete from trainset.
    :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
    '''

    path = os.path.join(root_dir, 'date_data.csv')
    total_df = pd.read_csv(path)

    # ignore warnings
    np.warnings.filterwarnings('ignore')

    df2018 = total_df[total_df['year'] == 2018]
    df2019 = total_df[total_df['year'] == 2019]

    if train_by_destination:
        train_dataframe, test_dataframe, y_train, y_test = train_test_split(total_df, total_df['destination'], test_size=0.3,
                                                              stratify=total_df['destination'], random_state=42)
    else:
        train_dataframe = df2018
        test_dataframe = df2019
        total_df = total_df[total_df['year'] != 2020]
    print(f"len(total): {len(total_df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
    return total_df, train_dataframe, test_dataframe,


class TourDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 train: bool):
        super(TourDataset, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train

        self.users, self.items = self._negative_sampling()
        print(f'len users:{self.users.shape}')
        print(f'len items:{self.items.shape}')


    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)

    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''

        # self.items[index][0]: positive feedback
        # self.items[index][1]: negative feedback
        if self.train:
            return self.users[index][0],  self.users[index][1], self.users[index][2], self.users[index][3], self.users[index][4], self.items[index][0], self.items[index][1]
        else:
            return self.users[index][1], self.users[index][1], self.users[index][2], self.users[index][3], self.users[index][4], self.items[index]

    def _negative_sampling(self):
        '''
        sampling one positive feedback per one negative feedback
        :return: dataframe
        '''
        print('Extract samples...')
        df = self.df
        total_df = self.total_df
        users_list, items_list = [], []
        all_destinations = total_df['itemid'].unique()

        # negative feedback dataset ratio
        if self.train:
            ng_ratio = 1
        else:
            ng_ratio = 9

        for i in df['userid'].unique():
            tmp = df.loc[df['userid'].isin([i])]
            med = tmp['congestion_1'].median()
            pos_item_set = zip(tmp['year'],
                                tmp['userid'],
                                tmp['age'],
                                tmp['dayofweek'],
                                tmp['sex'],
                                tmp.loc[tmp['congestion_1'] >= med, 'itemid'])

            neg_items = list(tmp.loc[tmp['congestion_1'] < med, 'itemid'])
            nil = list(set(all_destinations)-set(tmp.loc[tmp['congestion_1'] >= med, 'itemid'])-set(neg_items))
            neg_items += nil

            for year, uid, a, d, s, iid in pos_item_set:
                # positive instance
                item=[]
                if not self.train:
                    items_list.append(iid)
                    users_list.append([year, uid, a, d, s])
                else:
                    item.append(iid)

                for k in range(ng_ratio):
                    # negative instance
                    i = 0
                    negative_item = neg_items.pop()

                    if self.train:
                        item.append(negative_item)
                    else:
                        items_list.append(negative_item)
                        users_list.append([year, uid, a, d, s])

                if self.train:
                    items_list.append(item)
                    users_list.append([year, uid, a, d, s])
        print('Sampling ended!')
        return torch.LongTensor(users_list), torch.Longensor(items_list)

