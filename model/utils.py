import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class Preprocess(object):
    def __init__(self, root_dir: str,
                 train_by_destination: bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        self.root_dir = root_dir
        self.train_by_destination = train_by_destination
        self.df_raw = self.load_preprocess_data()
        self.user_dict = None
        self.item_dict = None
        self.date_dict = None

    def load_preprocess_data(self):
        root_dir = self.root_dir
        path = os.path.join(root_dir, 'Datasets_v5.0.txt')
        df_raw = pd.read_csv(path, sep='|')

        # consider congestion as preference
        df_raw[['congestion_1', 'congestion_2']] = 1 / df_raw[['congestion_1', 'congestion_2']]
        df_raw = df_raw.drop(columns=['total_num', 'area', 'date365'])
        df_raw['date'] = pd.to_datetime(df_raw['date'].astype('str'))

        # reshape data seperated by time zone into one day
        df_raw = pd.pivot_table(df_raw, index=['date', 'destination', 'dayofweek', 'sex', 'age'],
                               aggfunc={'congestion_1': 'sum',
                                        'congestion_2': 'sum',
                                        'visitor': 'sum'})
        df_raw = df_raw.reset_index()

        # seperate year and month-day data to use as features
        df_raw['year'] = df_raw['date'].dt.strftime('%y')
        df_raw['month-day'] = df_raw['date'].dt.strftime('%m%d')

        # Robust scaler to transform data with many outliers to dense data
        scaler = RobustScaler()
        df_raw[['visitor', 'congestion_1', 'congestion_2']] =\
            pd.DataFrame(scaler.fit_transform(df_raw[['visitor', 'congestion_1', 'congestion_2']]))

        # shift data to eliminate negative values and to use as explicit feedback
        df_raw['visitor'] = df_raw['visitor'] + df_raw['visitor'].min()
        df_raw['congestion_1'] = df_raw['congestion_1'] + df_raw['congestion_1'].min()
        df_raw['congestion_2'] = df_raw['congestion_2'] + df_raw['congestion_2'].min()
        return df_raw

    def map_userid(self):
        train_by_destination = self.train_by_destination

        if train_by_destination:
            df = self.df_raw
        else:
            df = self.df_raw.loc[self.df_raw['year'] != '20']

        # use age, sex, date as user Id
        def merge_cols():
            merged = pd.Series(df['age'].apply(str) + df['sex'].apply(str) + df['month-day'].apply(str))
            user_map = {item: i for i, item in enumerate(np.sort(merged.unique()))}
            item_map = {item: i for i, item in enumerate(np.sort(df['destination'].unique()))}
            date_map = {item: i for i, item in enumerate(np.sort(df['month-day'].unique()))}
            return merged, user_map, item_map, date_map

        vec_merge = np.vectorize(merge_cols)
        merged, user_map, item_map, date_map = vec_merge()

        # map user Id to value in dataframe
        def map_func(a, b, c):
            return user_map[a], item_map[b], date_map[c]

        np.warnings.filterwarnings('ignore')
        vec_func = np.vectorize(map_func)
        df.loc[:, 'userid'], df.loc[:, 'itemid'], df.loc[:, 'dateid'] =\
            vec_func(merged, df['destination'], df['month-day'])

        self.user_dict = user_map
        self.item_dict = item_map
        self.date_dict = date_map
        return df

    def split_train_test(self):
        total_df = self.map_userid()
        train_by_destination = self.train_by_destination

        # ignore warnings
        np.warnings.filterwarnings('ignore')
        total_df[['year', 'month-day']] = total_df[['year', 'month-day']].apply(np.int64)
        df_18 = total_df.loc[total_df['year'] == 18]
        df_19 = total_df.loc[total_df['year'] == 19].sample(frac=0.3, replace=False)

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
                 train: bool,
                 rating:str):
        super(TourDataset, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train
        self.rating = rating

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
        # train: year, uid, a, d, s, w, pos, neg
        # test:  year, uid, a, d, s, r, w, pos
        if self.train:
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.users[index][5], \
                   self.items[index][0], self.items[index][1]
        else:
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.users[index][5], self.users[index][6],\
                   self.items[index]

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
            ng_ratio = 24

        for i in df['userid'].unique():
            tmp = df.loc[df['userid'].isin([i])]
            quarter = tmp[self.rating].quantile(q=0.25)
            pos_item_set = zip(tmp['year'],
                               tmp['userid'],
                               tmp['age'],
                               tmp['dateid'],
                               tmp['sex'],
                               tmp[self.rating],
                               tmp['dayofweek'],
                               tmp.loc[tmp[self.rating] >= quarter, 'itemid'])

            neg_items = np.setxor1d(all_destinations, tmp.loc[tmp[self.rating] >= quarter, 'itemid'])

            for year, uid, a, d, s, r, w, iid in pos_item_set:
                tmp_negs = neg_items.copy()
                # positive instance
                item = []
                if not self.train:
                    items_list.append(iid)
                    users_list.append([year, uid, a, d, s, r, w])
                else:
                    item.append(iid)

                # negative instance
                negative_item = np.random.choice(tmp_negs, ng_ratio, replace=False)

                if self.train:
                    item += negative_item.tolist()
                else:
                    items_list += negative_item.tolist()
                    for _ in range(ng_ratio):
                        users_list.append([year, uid, a, d, s, r, w])

                if self.train:
                    items_list.append(item)
                    users_list.append([year, uid, a, d, s, w])
        print('Sampling ended!')
        return torch.LongTensor(users_list), torch.LongTensor(items_list)
