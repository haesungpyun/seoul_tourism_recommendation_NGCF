import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from datetime import datetime
from parsers import args


class Preprocess(object):
    def __init__(self, root_dir: str,
                 train_by_destination: bool,
                 folder_path: str,
                 rating_col:str,
                 scaler: str,
                 save_data: bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        self.root_dir = root_dir
        self.train_by_destination = train_by_destination
        self.folder_path = folder_path

        self.rating_col = rating_col
        self.save_data = save_data
        self.scaler = scaler

        self.user_dict = None
        self.item_dict = None
        self.num_dict = None

        self.df_raw = self.load_preprocess_data()

    # 데이터를 불러오고 preprocessing 진행, 일자별 타임존을 무시하고 합침, datetime으로 변경
    def load_preprocess_data(self):
        root_dir = self.root_dir
        path = os.path.join(root_dir, 'Datasets_v5.0.txt')
        df_raw = pd.read_csv(path, sep='|').sample(100)

        # consider congestion as preference
        df_raw = df_raw.drop(columns=['total_num', 'area', 'date365'])
        df_raw['date'] = pd.to_datetime(df_raw['date'].astype('str'))

        # reshape data seperated by time zone into one day
        df_raw = pd.pivot_table(df_raw, index=['date', 'destination', 'dayofweek', 'sex', 'age'],
                                aggfunc={'visitor': 'sum'})
        df_raw = df_raw.reset_index()

        # seperate year and month-day data to use as features
        df_raw['year'] = df_raw['date'].dt.strftime('%y')
        df_raw['month'] = df_raw['date'].dt.strftime('%m')
        df_raw['day'] = df_raw['date'].dt.strftime('%d')
        df_raw['month-day'] = pd.DataFrame(df_raw['month'].apply(str) + df_raw['day'].apply(str))
        df_raw[['year', 'month', 'day']] = df_raw[['year', 'month', 'day']].apply(np.int64)
        return df_raw

    # 각 user(나이, 성별, 방문 월, 일)별 고유 user id, 각 관광지 별 item id 생성 후 mapping
    def map_ids(self, df_raw):
        np.warnings.filterwarnings('ignore')
        train_by_destination = self.train_by_destination

        if train_by_destination:
            df = df_raw
        else:
            df = df_raw.loc[df_raw['year'] != 20]

        # use age, sex, date as user Id
        # user, item id 생성 후 mapping
        def merge_cols():
            merged = pd.Series(df['age'].apply(str) + df['sex'].apply(str) + df['month-day'].apply(str))
            user_map = {item: i for i, item in enumerate(np.sort(merged.unique()))}
            item_map = {item: i for i, item in enumerate(np.sort(df['destination'].unique()))}
            return merged, user_map, item_map

        vec_merge = np.vectorize(merge_cols)
        merged, user_map, item_map = vec_merge()

        # map user Id to value in dataframe
        def map_func(a, b):
            return user_map[a], item_map[b]

        vec_func = np.vectorize(map_func)
        df.loc[:, 'userid'], df.loc[:, 'itemid'] = vec_func(merged, df['destination'])
        self.user_dict = user_map
        self.item_dict = item_map

        if self.save_data:
            d1 = datetime.now()
            PATH = os.path.join(self.folder_path, f'user_dict_implicit_{args.epoch}_{args.batch_size}_{args.lr}_{args.emb_ratio}_{args.scaler}_{d1.month}_{d1.day}_{d1.hour}_{d1.minute}' + '.pkl')
            with open(PATH, 'wb') as f:
                pickle.dump(self.user_dict, f)
            PATH = os.path.join(self.folder_path, f'item_dict_implicit_{args.epoch}_{args.batch_size}_{args.lr}_{args.emb_ratio}_{args.scaler}_{d1.month}_{d1.day}_{d1.hour}_{d1.minute}' + '.pkl')
            with open(PATH, 'wb') as f:
                pickle.dump(self.item_dict, f)
            print('User, Item data Saved!')
        return df

    # scaling 후 모든 값이 0 이상이도록 min 값만큼 shift
    # 각 user의 rating 기준에 따라서(1/4, 1분위) 그 이하는 0으로 그 이상은 rating(visitor) 그대로 유지
    # explicit한 data의 성질은 살리고 bpr loss, negative sampling의 장점을 살리기 위해서
    #
    def scale_implicit(self):
        total_df = self.map_ids(self.df_raw)

        # PowerTransformer or StandardScaler to transform data with many outliers to dense data
        if self.scaler == 'power':
            scaler = PowerTransformer()
        else:
            scaler = StandardScaler()

        total_df[[self.rating_col]] = pd.DataFrame(scaler.fit_transform(total_df[[self.rating_col]]))
        # shift data to eliminate negative values and to use as explicit feedback
        v_min = np.abs(total_df[self.rating_col].min())
        total_df[self.rating_col] = total_df[self.rating_col] + v_min

        for userid in total_df['userid'].unique():
            tmp = total_df.loc[total_df['userid'].isin([userid])]
            quarter = tmp[self.rating_col].quantile(q=0.25)
            neg_tmp = tmp.loc[tmp[self.rating_col] < quarter]
            total_df.loc[neg_tmp.index, self.rating_col] = 0
        return total_df

    # train, test data 분리, test에 속하지 않은 0.7의 19년도 데이터
    # train data에 추가
    def split_train_test(self):
        total_df = self.scale_implicit()
        train_by_destination = self.train_by_destination

        # ignore warnings
        np.warnings.filterwarnings('ignore')

        df_18 = total_df.loc[total_df['year'] == 18]

        train_19 = total_df.loc[total_df['year'] == 19]
        df_19 = train_19.sample(frac=0.3, replace=False)
        train_19 = pd.concat((train_19, df_19), axis=0).drop_duplicates(keep=False)

        df_18 = pd.concat((df_18, train_19), axis=0)

        if train_by_destination:
            train_dataframe, test_dataframe, y_train, y_test = train_test_split(total_df, total_df['destination'],
                                                                                test_size=0.3,
                                                                                stratify=total_df['destination'],
                                                                                random_state=42)
        else:
            train_dataframe = df_18
            test_dataframe = df_19

        print(f"len(total): {len(total_df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")

        self.num_dict = {'user': total_df['userid'].nunique(),
                         'item': total_df['itemid'].nunique(),
                         'sex': total_df['sex'].max() + 1,
                         'age': total_df['age'].max() + 1,
                         'month': total_df['month'].max() + 1,
                         'day': total_df['day'].max() + 1,
                         'dayofweek': total_df['dayofweek'].max() + 1}
        print(self.num_dict)
        if self.save_data:
            PATH = os.path.join(self.folder_path, f'num_dict' + '.pkl')
            with open(PATH, 'wb') as f:
                pickle.dump(self.num_dict, f)
        return total_df, train_dataframe, test_dataframe, self.num_dict


class TourDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 train: bool,
                 rating_col: str):
        super(TourDataset, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train
        self.rating_col = rating_col

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
        # train: year, uid, a, s, m, d, dow, pos, neg
        if self.train:
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.users[index][5], self.users[index][6], \
                   self.items[index][0], self.items[index][1]
        # test:  year, uid, a, s, m, d, dow, r, pos
        else:
            return self.users[index][0], self.users[index][1], self.users[index][2], self.users[index][3], \
                   self.users[index][4], self.users[index][5], self.users[index][6], self.users[index][7], \
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

        # 각 user id 별로 negative sampling을 진행
        # user id가 방문하지 않은 item id, 1/4(1분위 이하)를 negative로 생각하여 sampling
        # train 시, 1개의 positive, negative
        # test 시, 1개의 positive, ng_ratio 개의 negative
        for userid in df['userid'].unique():
            tmp = df.loc[df['userid'].isin([userid])]
            pos_tmp = tmp.loc[tmp[self.rating_col] > 0]
            pos_items = pos_tmp.loc[:, 'itemid']
            neg_items = np.setxor1d(all_destinations, pos_items)

            pos_item_set = zip(pos_tmp['year'],
                               pos_tmp['userid'],
                               pos_tmp['age'],
                               pos_tmp['sex'],
                               pos_tmp['month'],
                               pos_tmp['day'],
                               pos_tmp['dayofweek'],
                               pos_tmp[self.rating_col],
                               pos_tmp['itemid'])

            for year, uid, a, s, m, d, dow, r, iid in pos_item_set:
                tmp_negs = neg_items.copy()
                # positive instance
                item = []
                if not self.train:
                    items_list.append(iid)
                    users_list.append([year, uid, a, s, m, d, dow, r])

                else:
                    item.append(iid)

                # negative instance
                negative_item = np.random.choice(tmp_negs, ng_ratio, replace=False)

                if self.train:
                    item += negative_item.tolist()
                else:
                    items_list += negative_item.tolist()
                    for _ in range(ng_ratio):
                        users_list.append([year, uid, a, s, m, d, dow, r])

                if self.train:
                    items_list.append(item)
                    users_list.append([year, uid, a, s, m, d, dow])
        print('Sampling ended!')
        return torch.LongTensor(users_list), torch.LongTensor(items_list),
