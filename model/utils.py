import torch as t
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


def split_train_test(root_dir, label_col) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    pick each unique userid row, and add to the testset, delete from trainset.
    :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
    '''
    path = os.path.join(root_dir, 'scale_data.csv')
    total_df = pd.read_csv(path)
    total_df = total_df
    train_dataframe = total_df
    test_dataframe = None
    for i in range(1):
        tmp_dataframe = train_dataframe.sample(frac=1).drop_duplicates(['useridx'])
        test_dataframe = pd.concat([tmp_dataframe, test_dataframe])
        tmp_dataframe2 = pd.concat([train_dataframe, tmp_dataframe])
        train_dataframe = tmp_dataframe2.drop_duplicates(keep=False)

    # explicit feedback -> implicit feedback
    # ignore warnings
    np.warnings.filterwarnings('ignore')
    # positive feedback (interaction exists)
    train_dataframe.loc[:, 'congestion_1'] = 1
    test_dataframe.loc[:, 'congestion_1'] = 1

    test_dataframe = test_dataframe.sort_values(by=['date', 'time'], axis=0)
    train_dataframe = train_dataframe.sort_values(by=['date', 'time'], axis=0)
    print(f"len(total): {len(total_df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
    return total_df, train_dataframe, test_dataframe,


class TourDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 label_col: str,
                 train: bool):
        super(TourDataset, self).__init__()

        self.df = df
        self.total_df = total_df
        self.label_col = label_col
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
            return self.users[index][0], self.users[index][1:], self.items[index][0], self.items[index][1]
        else:
            return self.users[index][0], self.users[index][1:], self.items[index]

    def _negative_sampling(self):
        '''
        sampling one positive feedback per one negative feedback
        :return: dataframe
        '''
        df = self.df
        total_df = self.total_df
        users_list, items_list = [], []
        user_item_set = zip(df['dateidx'],
                            df['useridx'],
                            df['age'],
                            df['dayofweek'],
                            df['time'],
                            df['sex'],
                            df['itemidx'])

        total_user_item_set = zip(total_df['dateidx'],
                                  total_df['useridx'],
                                  total_df['age'],
                                  total_df['dayofweek'],
                                  total_df['time'],
                                  total_df['sex'],
                                  total_df['itemidx'])
        all_destinations = total_df['itemidx'].unique()

        # negative feedback dataset ratio
        if self.train:
            ng_ratio = 1
        else:
            ng_ratio = 99

        for date, uid, a, d, t, s, iid in user_item_set:
            # positive instance
            visit = []
            item = []
            if not self.train:
                items_list.append(iid)
                users_list.append([date, uid, a, d, t, s])
            else:
                item.append(iid)

            for k in range(ng_ratio):
                # negative instance
                negative_item = np.random.choice(all_destinations)
                # check if item and user has interaction, if true then set new value from random
                while (date, uid, a, d, t, s, negative_item) in total_user_item_set or negative_item in visit:
                    negative_item = np.random.choice(all_destinations)

                if self.train:
                    item.append(negative_item)
                    visit.append(negative_item)
                else:
                    items_list.append(negative_item)
                    visit.append(negative_item)
                    users_list.append([date, uid, a, d, t, s])

            if self.train:
                items_list.append(item)
                users_list.append([date, uid, a, d, t, s])

        return t.tensor(users_list), t.tensor(items_list)


        """
        def _split_data_labels(self):
            df = self.total_df
            user_col_list = [col for col in df.columns if
                             col not in ['date', 'destination', 'congestion_1', 'congestion_2']]
            dates, users, items, labels = df['date'], df[user_col_list], df['destination'], df[self.label_col]
    
            dates = dates.values.tolist()
            users = users.values.tolist()
            items = items.values.tolist()
            labels = labels.values.tolist()
    
            print(f'len(dates):{len(dates)}, len(users):{len(users)}, len(items):{len(items)}, len(labels):{len(labels)}')
    
            return t.FloatTensor(dates), t.FloatTensor(users), t.LongTensor(items), t.FloatTensor(labels)
    
        """