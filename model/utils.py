import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Preprocess(object):
    def __init__(self, root_dir: str,
                 train_by_destination: bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        self.root_dir = root_dir
        self.train_by_destination = train_by_destination
        self.df_raw = self.load_data()
        self.user_dict = None
        self.item_dict = None
        warnings.filterwarnings('ignore')

    def load_data(self):
        root_dir = self.root_dir

        path = os.path.join(root_dir, 'date_data.csv')
        return pd.read_csv(path)

    def map_userid(self):
        train_by_destination = self.train_by_destination

        if train_by_destination:
            df = self.df_raw
        else:
            df = self.df_raw.loc[self.df_raw['year'] != 20]

        def merge_cols():
            merged = pd.Series(df['month-day'].apply(str)+df['dayofweek'].apply(str) +\
                               df['sex'].apply(str) + df['age'].apply(str))
            user_map = {item: i for i, item in enumerate(np.sort(merged.unique()))}
            item_map = {item: i for i, item in enumerate(np.sort(df['destination'].unique()))}
            date_map = {item: i for i, item in enumerate(np.sort(df['month-day'].unique()))}
            return merged, user_map, item_map, date_map

        vec_merge = np.vectorize(merge_cols)
        merged, user_map, item_map, date_map = vec_merge()

        def map_func(a, b, c):
            return user_map[a], item_map[b], date_map[c]

        vec_func = np.vectorize(map_func)
        df.loc[:, 'userid'], df.loc[:, 'itemid'], df.loc[:, 'dateid'] =\
            vec_func(merged, df['destination'],df['month-day'])

        self.user_dict = user_map
        self.item_dict = item_map
        return df

    def split_train_test(self):
        total_df = self.map_userid()
        train_by_destination = self.train_by_destination
        df_18 = total_df.loc[total_df['year'] == 18]
        df_19 = total_df.loc[total_df['year'] == 19]

        if train_by_destination:
            train_dataframe, test_dataframe, y_train, y_test = train_test_split(total_df, total_df['destination'],
                                                                                test_size=0.3,
                                                                                stratify=total_df['destination'],
                                                                                random_state=42)
        else:
            train_dataframe = df_18
            test_dataframe = df_19

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
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.items[index][0], self.items[index][1]
        else:
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.items[index]

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
            quarter = tmp['congestion_1'].quantile(q=0.25)
            pos_item_set = zip(tmp['year'],
                               tmp['userid'],
                               tmp['age'],
                               tmp['dayofweek'],
                               tmp['sex'],
                               tmp.loc[tmp['congestion_1'] >= quarter, 'itemid'])

            neg_items = np.setxor1d(all_destinations, tmp.loc[tmp['congestion_1'] >= quarter, 'itemid'])

            for year, uid, a, d, s, iid in pos_item_set:
                tmp_negs = neg_items.copy()
                # positive instance
                item = []
                if not self.train:
                    items_list.append(iid)
                    users_list.append([year, uid, a, d, s])
                else:
                    item.append(iid)

                for k in range(ng_ratio):
                    # negative instance
                    negative_item = np.random.choice(tmp_negs)
                    tmp_negs = np.delete(tmp_negs, np.where(tmp_negs == negative_item))

                    if self.train:
                        item.append(negative_item)
                    else:
                        items_list.append(negative_item)
                        users_list.append([year, uid, a, d, s])

                if self.train:
                    items_list.append(item)
                    users_list.append([year, uid, a, d, s])
        print('Sampling ended!')
        return torch.LongTensor(users_list), torch.LongTensor(items_list)
